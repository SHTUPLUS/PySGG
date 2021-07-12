import copy

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysgg.modeling.make_layers import make_fc
from pysgg.utils.comm import get_rank
from pysgg.config import cfg
from pysgg.modeling.roi_heads.relation_head.model_msg_passing import (
    PairwiseFeatureExtractor,
)
from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    make_relation_confidence_aware_module,
)
from pysgg.structures.boxlist_ops import squeeze_tensor


class MessagePassingUnit_v2(nn.Module):
    def __init__(self, input_dim, filter_dim=128):
        super(MessagePassingUnit_v2, self).__init__()
        self.w = nn.Linear(input_dim, filter_dim, bias=True)
        self.fea_size = input_dim
        self.filter_size = filter_dim

    def forward(self, unary_term, pair_term):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        # print '[unary_term, pair_term]', [unary_term, pair_term]
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = torch.sigmoid(gate.sum(1))
        # print 'gate', gate
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])

        return output, gate


def reverse_sigmoid(x):
    new_x = x.clone()
    new_x[x > 0.999] = x[x > 0.999] - (x[x > 0.999].clone().detach() - 0.999)
    new_x[x < 0.001] = x[x < 0.001] + (-x[x < 0.001].clone().detach() + 0.001)
    return torch.log((new_x) / (1 - (new_x)))


class MessagePassingUnit_v1(nn.Module):
    def __init__(self, input_dim, filter_dim=64):
        """

        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnit_v1, self).__init__()
        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )

        self.fea_size = input_dim
        self.filter_size = filter_dim


    def forward(self, unary_term, pair_term, aux_gate=None):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        paired_feats = torch.cat([unary_term, pair_term], 1)

        gate = torch.sigmoid(self.w(paired_feats))
        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes

        output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])

        return output, gate



class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout):
        super(MessageFusion, self).__init__()
        self.wih = nn.Linear(input_dim, input_dim, bias=True)
        self.whh = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output



class MSDNContext(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        hidden_dim=1024,
        num_iter=2,
        dropout=False,
        gate_width=128,
        use_kernel_function=False,
    ):
        super(MSDNContext, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.update_step = num_iter

        if self.update_step < 1:
            print(
                "WARNING: the update_step should be greater than 0, current: ",
                +self.update_step,
            )
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        self.pooling_dim = self.pairwise_feature_extractor.pooling_dim


        self.rel_aware_module_type = (
            self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        )

        self.num_rel_cls = self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.relness_weighting_mp = False
        self.gating_with_relness_logits = False
        self.filter_the_mp_instance = False
        self.relation_conf_aware_models = None
        self.apply_gt_for_rel_conf = False

        self.mp_pair_refine_iter = 1

        self.graph_filtering_method = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        )

        self.vail_pair_num = cfg.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.MP_VALID_PAIRS_NUM


        # decrease the dimension before mp
        self.obj_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )
        self.rel_downdim_fc = nn.Sequential(
            make_fc(self.pooling_dim, self.hidden_dim),
            nn.ReLU(True),
        )

        self.obj_pair2rel_fuse = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * 2),
            make_fc(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
        )

        self.padding_feature = nn.Parameter(
            torch.zeros((self.hidden_dim)), requires_grad=False
        )

        if use_kernel_function:
            MessagePassingUnit = MessagePassingUnit_v2
        else:
            MessagePassingUnit = MessagePassingUnit_v1


        self.gate_sub2pred = MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_obj2pred =MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_pred2sub = MessagePassingUnit(self.hidden_dim, gate_width)
        self.gate_pred2obj = MessagePassingUnit(self.hidden_dim, gate_width)


        self.object_msg_fusion = MessageFusion(self.hidden_dim, dropout)
        self.pred_msg_fusion = MessageFusion(self.hidden_dim, dropout)

        self.forward_time = 0



    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs, ):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:
            ALL RETURN THINGS ARE BATCH-WISE CONCATENATED

            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            selected_relness,
                the relatness score for selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
            selected_rel_prop_pairs_idx
                the relationship proposal id that selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
        """
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]
        rel_prop_pairs_relness_batch = []

        for idx, (prop, rel_ind_i) in enumerate(
            zip(
                proposals,
                rel_pair_idxs,
            )
        ):

            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)
        rel_inds_batch_cat = torch.cat(rel_inds_batch_cat, 0)

        subj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )
        obj_pred_map = (
            rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0])
            .fill_(0)
            .float()
            .detach()
        )


        # or all relationship pairs
        selected_rel_prop_pairs_idx = torch.arange(
            len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device
        )
        subj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1)

        return (
            rel_inds_batch_cat,
            subj_pred_map,
            obj_pred_map,
            selected_rel_prop_pairs_idx,
        )

    # Here, we do all the operations out of loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    def prepare_message(
        self,
        target_features,
        source_features,
        select_mat,
        gate_module,
    ):
        """
        generate the message from the source nodes for the following merge operations.

        Then the message passing process can be
        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_inst, rel_pair_num)
        :param gate_module:
        :param relness_scores: (num_rel, )
        :param relness_logit (num_rel, num_rel_category)

        :return: messages representation: (num_inst, dim)
        """
        feature_data = []

        if select_mat.sum() == 0:
            temp = torch.zeros(
                (target_features.size()[1:]),
                requires_grad=True,
                dtype=target_features.dtype,
                device=target_features.dtype,
            )
            feature_data = torch.stack(temp, 0)
        else:
            transfer_list = (select_mat > 0).nonzero()
            source_indices = transfer_list[:, 1]
            target_indices = transfer_list[:, 0]
            source_f = torch.index_select(source_features, 0, source_indices)
            target_f = torch.index_select(target_features, 0, target_indices)


            transferred_features, weighting_gate = gate_module(target_f, source_f)
            aggregator_matrix = torch.zeros(
                (target_features.shape[0], transferred_features.shape[0]),
                dtype=weighting_gate.dtype,
                device=weighting_gate.device,
            )

            for f_id in range(target_features.shape[0]):
                if select_mat[f_id, :].data.sum() > 0:
                    # average from the multiple sources
                    feature_indices = squeeze_tensor(
                        (transfer_list[:, 0] == f_id).nonzero()
                    )  # obtain source_relevant_idx
                    # (target, source_relevant_idx)
                    aggregator_matrix[f_id, feature_indices] = 1
            # (target, source_relevant_idx) @ (source_relevant_idx, feat-dim) => (target, feat-dim)
            aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
            avg_factor = aggregator_matrix.sum(dim=1)
            vaild_aggregate_idx = avg_factor != 0
            avg_factor = avg_factor.unsqueeze(1).expand(
                avg_factor.shape[0], aggregate_feat.shape[1]
            )
            aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

            feature_data = aggregate_feat
        return feature_data

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs):
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.hidden_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)

        obj_pair_feat4rel_rep = torch.cat(
            (head_rep[rel_pair_idxs[:, 0]], tail_rep[rel_pair_idxs[:, 1]]), dim=-1
        )

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(
            obj_pair_feat4rel_rep
        )  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(
        self,
        inst_features,
        rel_union_features,
        proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None,
    ):
        """

        :param inst_features: instance_num, pooling_dim
        :param rel_union_features:  rel_num, pooling_dim
        :param proposals: instance proposals
        :param rel_pair_inds: relaion pair indices list(tensor)
        :param rel_binarys: [num_prop, num_prop] the relatedness of each pair of boxes
        :return:
        """

        num_inst_proposals = [len(b) for b in proposals]

        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )

        refined_inst_features = augment_obj_feat
        refined_rel_features = rel_feats

        # build up list for massage passing process
        inst_feature4iter = self.obj_downdim_fc(augment_obj_feat)
        rel_feature4iter = self.rel_downdim_fc(rel_feats)

        valid_inst_idx = []
        if self.filter_the_mp_instance:
            for p in proposals:
                valid_inst_idx.append(p.get_field("pred_scores") > 0.1)

        if len(valid_inst_idx) > 0:
            valid_inst_idx = torch.cat(valid_inst_idx, 0)
        else:
            valid_inst_idx = torch.zeros(0)


        (
            batchwise_rel_pair_inds,
            subj_pred_map,
            obj_pred_map,
            selected_rel_prop_pairs_idx,
        ) = self._prepare_adjacency_matrix(proposals, rel_pair_inds)



        # graph module
        for t in range(self.update_step):
            param_idx = 0
            if not self.share_parameters_each_iter:
                param_idx = t
            """update object features pass message from the predicates to instances"""
            object_sub = self.prepare_message(
                inst_feature4iter[t],
                rel_feature4iter[t],
                subj_pred_map,
                self.gate_pred2sub[param_idx],
            )
            object_obj = self.prepare_message(
                inst_feature4iter[t],
                rel_feature4iter[t],
                obj_pred_map,
                self.gate_pred2obj[param_idx],
            )

            GRU_input_feature_object = (object_sub + object_obj) / 2.0
            inst_feature4iter = inst_feature4iter + self.object_msg_fusion[param_idx](
                    GRU_input_feature_object, inst_feature4iter
                )

            """update predicate features from entities features"""
            indices_sub = batchwise_rel_pair_inds[:, 0]
            indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1



            # obj to pred on all pairs
            feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
            feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
            phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
                rel_feature4iter[t], feat_sub2pred
            )
            phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
                rel_feature4iter[t], feat_obj2pred
            )
            GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
            rel_feature4iter.append(
                rel_feature4iter[t]
                + self.pred_msg_fusion[param_idx](
                    GRU_input_feature_phrase, rel_feature4iter[t]
                )
            )
            
        refined_inst_features = inst_feature4iter[-1]
        refined_rel_features = rel_feature4iter[-1]

        return (
            refined_inst_features,
            refined_rel_features,
        )


def build_msdn_model(cfg, in_channels):
    return MSDNContext(cfg, in_channels)
