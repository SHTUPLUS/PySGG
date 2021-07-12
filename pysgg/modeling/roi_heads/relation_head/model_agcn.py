# Graph R-CNN for scene graph generation from jwyang' codebase
# Re-implemented by us
import copy
import math

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.model_msg_passing import PairwiseFeatureExtractor
from pysgg.structures.boxlist_ops import squeeze_tensor


class GRCNN(nn.Module):
    # def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, cfg, in_channels, hidden_dim=1024):
        super(GRCNN, self).__init__()
        self.cfg = cfg
        self.dim = hidden_dim
        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.FEATURE_UPDATE_STEP
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.SCORES_UPDATE_STEP
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        # self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, in_channels)

        self.vail_pair_num = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.MP_VALID_PAIRS_NUM

        self.filter_the_mp_instance = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.MP_ON_VALID_PAIRS
        self.graph_filtering_method = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        # graph will only message passing on edges filtered by the rel pn structure

        self.mp_weighting = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.RELNESS_MP_WEIGHTING

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(
            cfg, in_channels)

        self.obj_embedding = nn.Sequential(
            nn.Linear(in_channels, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(in_channels, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        if self.feat_update_step > 0:
            self.gcn_collect_feat = GraphConvolutionCollectLayer(
                self.dim, self.dim)
            self.gcn_update_feat = GraphConvolutionUpdateLayer(
                self.dim, self.dim)

        self.obj_hidden_embedding = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.dim, num_classes_obj),
        )
        self.rel_hidden_embedding = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.dim, num_classes_pred),
        )

        if self.score_update_step > 0:
            self.gcn_collect_score = GraphConvolutionCollectLayer(
                num_classes_obj, num_classes_pred, att_head_num=1)
            self.gcn_update_score = GraphConvolutionUpdateLayer(
                num_classes_obj, num_classes_pred)

    def _get_map_idxs(self, num_proposals, rel_pair_idxs, relatedness):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:
            rel_inds,
                extent the instances pairing matrix to the batch wised (num_inst, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
        """
        rel_inds = []
        offset = 0
        rel_prop_pairs_relness_batch = []



        for idx, (prop_num, rel_ind_i) in enumerate(zip(num_proposals, rel_pair_idxs, )):
            if self.filter_the_mp_instance:
                assert relatedness is not None
                related_matrix = relatedness[idx]
                rel_prop_pairs_relness = related_matrix[rel_ind_i[:, 0],
                                                        rel_ind_i[:, 1]]
                # get the valid relation pair for the message passing
                rel_prop_pairs_relness_batch.append(rel_prop_pairs_relness)
            rel_ind_i = copy.deepcopy(rel_ind_i)

            rel_ind_i += offset
            offset += prop_num
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = rel_inds.new(
            sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(
            sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)

        obj_obj_map = torch.zeros(sum(num_proposals), sum(num_proposals),
                                  device=rel_pair_idxs[0].device).float()

        obj_obj_map[rel_inds[:, 0], rel_inds[:, 1]] = 1
        obj_obj_map[rel_inds[:, 1], rel_inds[:, 0]] = 1

        # only message passing on valid pairs
        selected_relness = None
        subj_pred_map_filtered = None
        obj_pred_map_filtered = None
        if len(rel_prop_pairs_relness_batch) != 0:

            subj_pred_map_filtered = rel_inds.new(
                sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()
            obj_pred_map_filtered = rel_inds.new(
                sum(num_proposals), rel_inds.shape[0]).fill_(0).float().detach()

            rel_prop_pairs_relness_batch_cat = torch.cat(
                rel_prop_pairs_relness_batch, 0)

            if self.graph_filtering_method == "rel_pn":
                _, \
                selected_rel_prop_pairs_idx = torch.sort(
                    rel_prop_pairs_relness_batch_cat, descending=True)

                selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[: self.vail_pair_num]

            elif self.graph_filtering_method == "gt":
                selected_rel_prop_pairs_idx = squeeze_tensor(
                    torch.nonzero(rel_prop_pairs_relness_batch_cat == 1))
            else:
                raise ValueError()

            subj_pred_map_filtered[rel_inds[selected_rel_prop_pairs_idx,
                                            0], selected_rel_prop_pairs_idx] = 1
            obj_pred_map_filtered[rel_inds[selected_rel_prop_pairs_idx,
                                           1], selected_rel_prop_pairs_idx] = 1
            selected_relness = rel_prop_pairs_relness_batch_cat[selected_rel_prop_pairs_idx]

            obj_obj_map = torch.zeros(sum(num_proposals), sum(num_proposals),
                                      device=rel_pair_idxs[0].device).float()

            obj_obj_map[rel_inds[selected_rel_prop_pairs_idx, 0],
                        rel_inds[selected_rel_prop_pairs_idx, 1]] = 1
            obj_obj_map[rel_inds[selected_rel_prop_pairs_idx, 1],
                        rel_inds[selected_rel_prop_pairs_idx, 0]] = 1

        return rel_inds, subj_pred_map, obj_pred_map, \
               subj_pred_map_filtered, obj_pred_map_filtered, \
               obj_obj_map, selected_relness

    def forward(self, inst_features, rel_union_features, proposals, rel_pair_inds, relatedness=None):

        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(inst_features, rel_union_features,
                                                                      proposals, rel_pair_inds, )

        num_inst_proposals = [len(b) for b in proposals]

        # all batch wise thing has been concate into one dim matrix
        batchwise_rel_pair_inds, subj_pred_map, obj_pred_map, \
        subj_pred_map_filtered, obj_pred_map_filtered, \
        obj_obj_map, selected_relness = self._get_map_idxs(num_inst_proposals, rel_pair_inds, relatedness)

        x_obj = self.obj_embedding(augment_obj_feat)
        x_pred = self.rel_embedding(rel_feats)

        '''feature level agcn'''
        obj_feats = [x_obj]
        pred_feats = [x_pred]

        for t in range(self.feat_update_step):
            # message from other objects
            update_feat_obj, vaild_mp_idx_obj = self.gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map,
                                                                 GraphConvolutionCollectLayer.INST2INST)

            # message from predicates to instances
            if subj_pred_map_filtered is not None:
                update_feat_rel_sub, vaild_mp_idx_rel_sub = self.gcn_collect_feat(obj_feats[t], pred_feats[t],
                                                                             subj_pred_map_filtered,
                                                                             GraphConvolutionCollectLayer.REL2SUB)
            else:
                update_feat_rel_sub, vaild_mp_idx_rel_sub = self.gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map,
                                                                             GraphConvolutionCollectLayer.REL2SUB)

            if obj_pred_map_filtered is not None:
                update_feat_rel_obj, vaild_mp_idx_rel_obj = self.gcn_collect_feat(obj_feats[t], pred_feats[t],
                                                                             obj_pred_map_filtered,
                                                                             GraphConvolutionCollectLayer.REL2OBJ)
            else:
                update_feat_rel_obj, vaild_mp_idx_rel_obj = self.gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map,
                                                                             GraphConvolutionCollectLayer.REL2OBJ)

            update_feat2ent_all = (update_feat_obj + update_feat_rel_sub + update_feat_rel_obj) / 3

            padded_next_stp_obj_feats = obj_feats[t].clone()
            update_feat = self.gcn_update_feat(
                torch.index_select(padded_next_stp_obj_feats, 0, vaild_mp_idx_rel_obj),
                torch.index_select(update_feat2ent_all, 0, vaild_mp_idx_rel_obj), 0
            )

            padded_next_stp_obj_feats[vaild_mp_idx_rel_obj] = update_feat
            obj_feats.append(padded_next_stp_obj_feats)

            # print(torch.nonzero(torch.isnan(padded_next_stp_obj_feats)))

            '''update predicate features'''
            source_obj_sub, vaild_mp_idx_obj_rel = self.gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(),
                                                                         GraphConvolutionCollectLayer.SUB2REL)
            source_obj_obj, vaild_mp_idx_sub_rel = self.gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(),
                                                                         GraphConvolutionCollectLayer.OBJ2REL)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2

            padded_next_stp_pred_feats = pred_feats[t].clone()
            update_feat = self.gcn_update_feat(
                torch.index_select(padded_next_stp_pred_feats, 0, vaild_mp_idx_obj_rel),
                torch.index_select(padded_next_stp_pred_feats, 0, vaild_mp_idx_obj_rel), 1
            )
            padded_next_stp_pred_feats[vaild_mp_idx_obj_rel] = update_feat
            pred_feats.append(padded_next_stp_pred_feats)

            # print(torch.nonzero(torch.isnan(padded_next_stp_pred_feats)))

        obj_class_logits = torch.cat(
            [proposal.get_field("predict_logits").detach() for proposal in proposals], 0)

        # 对于logits的prediction 没有分类的supervison 本质上就是个 2 layers stacked-AGCN
        # 这里可以把predicting 改到外面去 这样本质上和MSDN的graph model的流程基本保持一一致
        obj_class_logits = self.obj_hidden_embedding(obj_feats[-1])
        pred_class_logits = self.rel_hidden_embedding(pred_feats[-1])

        '''score level agcn'''
        obj_scores = [obj_class_logits]
        pred_scores = [pred_class_logits]

        for t in range(self.score_update_step):
            '''entities to entities'''
            update_feat_obj, vaild_mp_idx_obj = self.gcn_collect_score(obj_scores[t], obj_scores[t], obj_obj_map,
                                                                  GraphConvolutionCollectLayer.INST2INST)

            '''entities to predicates'''
            if subj_pred_map_filtered is not None:
                update_feat_rel_sub, vaild_mp_idx_rel_sub = self.gcn_collect_score(obj_scores[t], pred_scores[t],
                                                                              subj_pred_map_filtered,
                                                                              GraphConvolutionCollectLayer.REL2SUB)
            else:
                update_feat_rel_sub, vaild_mp_idx_rel_sub = self.gcn_collect_score(obj_scores[t], pred_scores[t],
                                                                              subj_pred_map,
                                                                              GraphConvolutionCollectLayer.REL2SUB)

            if obj_pred_map_filtered is not None:
                update_feat_rel_obj, vaild_mp_idx_rel_obj = self.gcn_collect_score(obj_scores[t], pred_scores[t],
                                                                              obj_pred_map_filtered,
                                                                              GraphConvolutionCollectLayer.REL2OBJ)
            else:
                update_feat_rel_obj, vaild_mp_idx_rel_obj = self.gcn_collect_score(obj_scores[t], pred_scores[t],
                                                                              obj_pred_map,
                                                                              GraphConvolutionCollectLayer.REL2OBJ)

            padded_next_stp_obj_feats = obj_scores[t].clone()
            update_feat2ent_all = (update_feat_obj + update_feat_rel_sub + update_feat_rel_obj) / 3
            update_feat = self.gcn_update_score(
                torch.index_select(padded_next_stp_obj_feats, 0, vaild_mp_idx_rel_obj),
                torch.index_select(update_feat2ent_all, 0, vaild_mp_idx_rel_obj), 0
            )
            padded_next_stp_obj_feats[vaild_mp_idx_rel_obj] = update_feat

            # print(torch.nonzero(torch.isnan(padded_next_stp_obj_feats)))

            obj_scores.append(padded_next_stp_obj_feats)

            '''update predicate logits'''
            source_obj_sub, vaild_mp_idx_obj_rel = self.gcn_collect_score(pred_scores[t], obj_scores[t],
                                                                          subj_pred_map.t(),
                                                                          GraphConvolutionCollectLayer.SUB2REL)
            source_obj_obj, vaild_mp_idx_sub_rel = self.gcn_collect_score(pred_scores[t], obj_scores[t],
                                                                          obj_pred_map.t(),
                                                                          GraphConvolutionCollectLayer.OBJ2REL)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2

            padded_next_stp_pred_feats = pred_scores[t].clone()
            update_feat = self.gcn_update_score(
                torch.index_select(padded_next_stp_pred_feats, 0, vaild_mp_idx_obj_rel),
                torch.index_select(source2rel_all, 0, vaild_mp_idx_obj_rel), 1
            )
            padded_next_stp_pred_feats[vaild_mp_idx_obj_rel] = update_feat

            # print(torch.nonzero(torch.isnan(padded_next_stp_pred_feats)))

            pred_scores.append(padded_next_stp_pred_feats)

        obj_class_logits = obj_scores[-1]
        pred_class_logits = pred_scores[-1]

        return obj_class_logits, pred_class_logits


def build_grcnn_model(cfg, in_channels):
    return GRCNN(cfg, in_channels)


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class UpdateUnit(nn.Module):
    def __init__(self, dim):
        super(UpdateUnit, self).__init__()

    def forward(self, target, source):
        assert target.size() == source.size(
        ), "source dimension must be equal to target dimension"
        update = target + source
        return update


def prepare_message(target, source, adj_matrix, trans_fc, att_module):
    # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
    source = F.relu(trans_fc(source))

    max_income_edge_num = int(torch.max(adj_matrix.sum(1)).item())
    att_mask = torch.ones(
        (target.shape[0], max_income_edge_num), device=source.device, dtype=torch.bool)
    vaild_mp_idx = []

    active_nodes_num = 0
    active_nodes_id = []
    active_nodes_indices = []
    for f_id in range(target.shape[0]):
        tmp_idx = squeeze_tensor(torch.nonzero(adj_matrix[f_id]))
        if len(tmp_idx) > 0:
            active_nodes_indices.append(tmp_idx)
            active_nodes_num += 1
            active_nodes_id.append(f_id)

    active_nodes_iter = 0
    selected_idx = []
    # generate idx in loops, do indexing at once
    for i, f_id in enumerate(active_nodes_id):
        # do 1 to N attention:
        #    multiple source nodes to single target node
        indices = active_nodes_indices[i]
        att_mask[f_id, torch.arange(len(indices))] = False
        padding = torch.zeros((max_income_edge_num - len(indices)), dtype=torch.long, device=indices.device)
        indices = torch.cat((indices, padding), dim=0)

        selected_idx.append(indices)
        active_nodes_iter += 1

    # reshape the indexs ad indexing
    selected_idx = torch.cat(selected_idx, dim=0)
    att_sources = source[selected_idx].reshape(active_nodes_iter, max_income_edge_num, -1)

    vaild_mp_idx = torch.Tensor(active_nodes_id).long().to(target.device)

    att_targets = target[vaild_mp_idx].unsqueeze(0).contiguous()  # 1, aggrate_num, dim
    att_sources = att_sources.transpose(0, 1).contiguous()  # source_num, aggrate_num, dim

    att_mask = att_mask[vaild_mp_idx]
    att_res, att_weight = att_module(query=att_targets, key=att_sources, value=att_sources,
                                     key_padding_mask=att_mask) # 1 to source
    att_res = squeeze_tensor(att_res)

    att_res_padded = torch.zeros((target.shape[0], att_res.shape[-1]),
                                 dtype=att_res.dtype, device=att_res.device, )

    att_res_padded[vaild_mp_idx] = att_res

    return att_res_padded, vaild_mp_idx


class GraphConvolutionCollectLayer(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    REL2SUB = 0
    REL2OBJ = 1
    SUB2REL = 2
    OBJ2REL = 3
    INST2INST = 4

    def __init__(self, dim_obj, dim_rel, att_head_num=4):
        super(GraphConvolutionCollectLayer, self).__init__()

        self.collect_units_fc = nn.ModuleList([
            make_fc(dim_rel, dim_obj),
            make_fc(dim_rel, dim_obj),
            make_fc(dim_obj, dim_rel),
            make_fc(dim_obj, dim_rel),
            make_fc(dim_obj, dim_obj),
        ])

        self.collect_units_att_module = nn.ModuleList([
            nn.MultiheadAttention(num_heads=att_head_num, embed_dim=dim_obj),
            nn.MultiheadAttention(num_heads=att_head_num, embed_dim=dim_obj),
            nn.MultiheadAttention(num_heads=att_head_num, embed_dim=dim_rel),
            nn.MultiheadAttention(num_heads=att_head_num, embed_dim=dim_rel),
            nn.MultiheadAttention(num_heads=att_head_num, embed_dim=dim_obj),
        ])

    def forward(self, target, source, adjc_matrix, unit_id):
        collection, vaild_mp_idx = prepare_message(target, source, adjc_matrix,
                                                   self.collect_units_fc[unit_id],
                                                   self.collect_units_att_module[unit_id])

        return collection, vaild_mp_idx


class GraphConvolutionUpdateLayer(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, dim_obj, dim_rel):
        super(GraphConvolutionUpdateLayer, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(UpdateUnit(dim_obj))  # obj from others
        self.update_units.append(UpdateUnit(dim_rel))  # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


def _attention(q, k, v, d_k, mask=None, dropout=None):
    """
    output: (batch_size, head_num, q_num, dim)
    scores: (batch_size, head_num, q_num, k_num)
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class _MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = make_fc(d_model, d_model)
        self.v_linear = make_fc(d_model, d_model)
        self.k_linear = make_fc(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = make_fc(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        # perform linear operation and split into h heads
        try:
            k = self.k_linear(k).view(-1, self.h, self.d_k)
            q = self.q_linear(q).view(-1, self.h, self.d_k)
            v = self.v_linear(v).view(-1, self.h, self.d_k)
        except RuntimeError:
            ipdb.set_trace()

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(0, 1)
        q = q.transpose(0, 1)
        v = v.transpose(0, 1)
        # calculate attention using function we will define next
        att_result, att_scores = _attention(
            q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = att_result.transpose(0, 1).contiguous().view(-1, self.d_model)

        output = self.out(concat)

        return squeeze_tensor(output)
