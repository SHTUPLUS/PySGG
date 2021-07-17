# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ipdb
import torch


from pysgg.modeling.roi_heads.relation_head.model_gpsnet import GPSNetContext
from torch import nn
from torch.nn import functional as F

from pysgg.config import cfg
from pysgg.data import get_dataset_statistics
from pysgg.modeling import registry
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    GGNNRelReason,
    InstanceFeaturesAugments,
    to_onehot,
)
from pysgg.modeling.roi_heads.relation_head.model_msdn import MSDNContext
from pysgg.modeling.roi_heads.relation_head.model_naive import (
    PairwiseFeatureExtractor,
)
from pysgg.modeling.utils import cat
from pysgg.structures.boxlist_ops import squeeze_tensor
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_msg_passing import IMPContext
from .model_transformer import TransformerContext
from .model_vctree import VCTreeLSTMContext
from .model_vtranse import VTransEFeature
from .model_bgnn import BGNNContext
from .model_agcn import GRCNN
from .rel_proposal_network.models import (
    make_relation_confidence_aware_module,
)
from .rel_proposal_network.loss import (
    FocalLossFGBGNormalization,
    RelAwareLoss,
)
from .utils_relation import layer_init, get_box_info, get_box_pair_info, obj_prediction_nms


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(
                roi_features, proposals, logger
            )
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(
                roi_features, proposals, logger
            )

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds
        ):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.context_layer = IMPContext(
            config,
            in_channels,
            hidden_dim=config.MODEL.ROI_RELATION_HEAD.IMP_MODULE.GRAPH_HIDDEN_DIM,
            num_iter=config.MODEL.ROI_RELATION_HEAD.IMP_MODULE.GRAPH_ITERATION_NUM,
        )

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

        self.init_classifier_weight()

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation

        obj_feats, rel_feats = self.context_layer(
            roi_features, proposals, union_features, rel_pair_idxs, logger
        )

        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.num_obj)
        else:
            obj_dists = self.obj_classifier(obj_feats)

        rel_dists = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MSDNPredictor")
class MSDNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MSDNPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.GRAPH_HIDDEN_DIM

        self.split_context_model4inst_rel = (
            config.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.SPLIT_GRAPH4OBJ_REL
        )
        if self.split_context_model4inst_rel:
            self.obj_context_layer = MSDNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.GRAPH_ITERATION_NUM,
            )
            self.rel_context_layer = MSDNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.GRAPH_ITERATION_NUM,
            )
        else:
            self.context_layer = MSDNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.GRAPH_ITERATION_NUM,
            )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
            )  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                rel_cls_logits
                + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        ## pre clser relpn supervision
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)

                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses


@registry.ROI_RELATION_PREDICTOR.register("BGNNPredictor")
class BGNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(BGNNPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM

        self.split_context_model4inst_rel = (
            config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SPLIT_GRAPH4OBJ_REL
        )
        if self.split_context_model4inst_rel:
            self.obj_context_layer = BGNNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
            self.rel_context_layer = BGNNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
        else:
            self.context_layer = BGNNContext(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            if self.mode == "sgdet":
                boxes_per_cls = cat(
                    [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
                )  # comes from post process of box_head
                # here we use the logits refinements by adding
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits = refined_obj_logits + obj_pred_logits
                if self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                refined_obj_pred_labels = obj_prediction_nms(
                    boxes_per_cls, obj_pred_logits, nms_thresh=0.5
                )
                obj_pred_labels = refined_obj_pred_labels
            else:
                _, obj_pred_labels = refined_obj_logits[:, 1:].max(-1)
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                rel_cls_logits
                + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        ## pre clser relpn supervision
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)

                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses


@registry.ROI_RELATION_PREDICTOR.register("GPSNetPredictor")
class GPSNetPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(GPSNetPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.GRAPH_HIDDEN_DIM

        self.context_layer = GPSNetContext(
            config,
            self.input_dim,
            hidden_dim=self.hidden_dim,
            num_iter=config.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.GRAPH_ITERATION_NUM,
        )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        self.focal_loss4pre_cls = FocalLossFGBGNormalization(alpha=1.0, gamma=0.0)
        # post classification
        self.rel_classifier = build_classifier(self.pooling_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.pooling_dim, self.num_obj_cls)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

        self.init_classifier_weight()

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
            )  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(
                pair_pred.long()
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses


@registry.ROI_RELATION_PREDICTOR.register("AGRCNNPredictor")
class AGRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(AGRCNNPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.GRAPH_HIDDEN_DIM

        # self.split_context_model4inst_rel = config.MODEL.ROI_RELATION_HEAD.GRCNN_MODULE.SPLIT_GRAPH4OBJ_REL

        self.context_layer = GRCNN(config, self.input_dim, self.hidden_dim)

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

        self.init_classifier_weight()

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        obj_feats, rel_feats = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys
        )
        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
            )  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(
                pair_pred.long()
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}
        return obj_pred_logits, rel_cls_logits, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(
                config, obj_classes, att_classes, rel_classes, in_channels
            )
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = make_fc(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = make_fc(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = build_classifier(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        self.init_classifier_weight()

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = make_fc(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.use_obj_recls_labels = config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def init_classifier_weight(self):
        self.rel_compress.reset_parameters()

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """
        vectors are batch concatenated features
        :param proposals:
        :param rel_pair_idxs:
        :param rel_labels: relation labels for computing the loss
        :param rel_binarys: relation proposal predict
        :param roi_features: object features
        :param union_features: union box ROI features of object in relation
        :param logger:
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(
                roi_features, proposals, logger
            )
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(
                roi_features, proposals, logger
            )

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        if not self.use_obj_recls_labels:
            obj_preds = [each.get_field("pred_labels") for each in proposals]

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds
        ):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # multiply the relation features from the pairs objects and the relationships
        # pairs union
        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if not self.use_obj_recls_logits:
            obj_dists = [each.get_field("predict_logits") for each in proposals]

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(
            config, obj_classes, rel_classes, statistics, in_channels
        )

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = make_fc(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = make_fc(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        # self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = make_fc(self.pooling_dim, self.num_rel_cls)
        # self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.uni_gate, xavier=True)
        # layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = make_fc(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(
            roi_features, proposals, rel_pair_idxs, logger
        )

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds
        ):
            prod_reps.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        # uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("NaivePredictor")
class NaivePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(NaivePredictor, self).__init__()
        self.cfg = config
        # todo no attribute recognition now
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        assert self.attribute_on == False

        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.freq_bias = None
        if cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.pairwise_obj_feat_updim_fc = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.output_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        self.rel_classifier = build_classifier(self.pooling_dim, self.num_rel_cls, bias=True)

        self.rel_pn_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.pooling_dim, self.pooling_dim)
        )
        self.rel_pn_module = PreClassifierRelPN(self.pooling_dim, False)
        self.pre_clser_loss_type = "bce"

        # initialize layer parameters
        layer_init(
            self.pairwise_obj_feat_updim_fc, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True
        )
        self.init_classifier_weight()

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim_linear = nn.Linear(
                config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim
            )
            layer_init(self.up_dim_linear, xavier=True)
        else:
            self.union_single_not_match = False

        self.obj_pair_feature_extractor = PairwiseFeatureExtractor(
            config, obj_classes, rel_classes, in_channels
        )

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                *[
                    nn.Linear(32, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.pooling_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_union_feat", torch.zeros(self.pooling_dim))

        # self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()

    def pair_feature_generate(
        self,
        roi_features,
        proposals,
        rel_pair_idxs,
        num_objs,
        obj_boxs,
        logger,
        ctx_average=False,
    ):
        """

        :param roi_features: object features
        :param proposals:  object proposals
        :param rel_pair_idxs: relation pair matrix
        :param num_objs: the object numbers of each images in batch
        :param obj_boxs: relation pair boxes geometry information list(Tensor(n_pairs, 12))
        :param logger:
        :param ctx_average:
        :return:
            obj_pair_feat4rel_rep:
                use object feat to represent the relationship feature (num_rels, hidden_dim)
            pair_preds_labels, relationship prediction labels (num_rels, )
            pair_bbox_geo_info, the pair boxes relative geometry information, (num_rels, 9)
            pair_obj_probs, the pair objects prediction probability distribution
            rel_binary_preds: relationship affinity prediction results
            obj_dist_prob, object prediction probability distribution
            pairwise_obj_feats_fused
            obj_dist_list
        """
        # encode context information
        (
            obj_dist_logits,
            obj_preds_labels,
            pairwise_obj_feats,
            rel_binary_preds,
        ) = self.obj_pair_feature_extractor(
            roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average
        )

        obj_dist_prob = F.softmax(obj_dist_logits, dim=-1)

        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(pairwise_obj_feats)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.edge_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds_labels = obj_preds_labels.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dist_logits.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_preds_labels = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds_labels, obj_boxs, obj_prob_list
        ):
            obj_pair_feat4rel_rep.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )
            pair_preds_labels.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
            pair_obj_probs.append(
                torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2)
            )
            pair_bboxs_info.append(
                get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]])
            )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        pair_preds_labels = cat(pair_preds_labels, dim=0)
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)

        obj_pair_feat4rel_rep = self.output_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return (
            obj_pair_feat4rel_rep,
            pair_preds_labels,
            pair_bbox_geo_info,
            pair_obj_probs,
            rel_binary_preds,
            obj_dist_prob,
            obj_dist_list,
        )

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param proposals: object [BoxList]
        :param rel_pair_idxs: the object pairing matrix list(Tensor)
        :param rel_labels:  GT rel_labels for loss computation list(Tensor)
        :param rel_binarys: GT rel_binary labels for loss computation list(Tensor)
        :param roi_features: object features list(Tensor)
        :param union_features: the union box features of each relationships list((Tensor)) or
                               list((tuple(Tensor))) contains visual features and  spatial box masks features
        :param logger:
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]



        assert len(num_rels) == len(num_objs)

        (
            obj_pair_feat4rel_rep,
            pair_preds_labels,
            pair_bbox_geo_info,
            pair_obj_probs,
            rel_binary_preds,
            obj_dist_prob,
            obj_dist_list,
        ) = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger
        )

        # generate the avg blank features for causalities comparison
        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                (
                    avg_post_ctx_rep,
                    _,
                    _,
                    avg_pair_obj_prob,
                    _,
                    _,
                    _,
                ) = self.pair_feature_generate(
                    roi_features,
                    proposals,
                    rel_pair_idxs,
                    num_objs,
                    obj_boxs,
                    logger,
                    ctx_average=True,
                )

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * spatial_conv_feats

        if self.spatial_for_vision:
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)

        # todo fusion calculate not TDE split style
        rel_dists = self.calculate_logits(
            union_features, obj_pair_feat4rel_rep, pair_preds_labels, use_label_dist=False
        )
        rel_dist_list = rel_dists.split(num_rels, dim=0)
        add_losses = {}

        # additional loss
        if self.effect_analysis:
            if self.training:
                # todo currently need auxiliary loss

                # untreated average feature
                if self.spatial_for_vision:
                    self.untreated_spt = self.moving_average(
                        self.untreated_spt, pair_bbox_geo_info
                    )
                if self.separate_spatial:
                    self.untreated_conv_spt = self.moving_average(
                        self.untreated_conv_spt, spatial_conv_feats
                    )
                self.avg_post_ctx = self.moving_average(
                    self.avg_post_ctx, obj_pair_feat4rel_rep
                )
                self.untreated_union_feat = self.moving_average(
                    self.untreated_union_feat, union_features
                )

            else:
                with torch.no_grad():
                    # untreated spatial
                    if self.spatial_for_vision:
                        avg_spt_rep = self.spt_emb(
                            self.untreated_spt.clone().detach().view(1, -1)
                        )
                    # untreated context
                    avg_ctx_rep = (
                        avg_post_ctx_rep * avg_spt_rep
                        if self.spatial_for_vision
                        else avg_post_ctx_rep
                    )
                    avg_ctx_rep = (
                        avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1)
                        if self.separate_spatial
                        else avg_ctx_rep
                    )
                    # untreated visual
                    avg_vis_rep = self.untreated_union_feat.clone().detach().view(1, -1)
                    # untreated category dist
                    avg_frq_rep = avg_pair_obj_prob

                if self.effect_type == "TDE":  # TDE of CTX
                    rel_dists = self.calculate_logits(
                        union_features, obj_pair_feat4rel_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
                elif self.effect_type == "NIE":  # NIE of FRQ
                    rel_dists = self.calculate_logits(
                        union_features, avg_ctx_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                elif self.effect_type == "TE":  # Total Effect
                    rel_dists = self.calculate_logits(
                        union_features, obj_pair_feat4rel_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                else:
                    assert self.effect_type == "none"
                    pass
                rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(
                0
            ).view(-1)
        return holder

    def calculate_logits(
        self,
        union_feat,
        pairwise_auged_obj_feats,
        frq_rep,
        use_label_dist=True,
        mean_ctx=False,
    ):
        """

        :param union_feat: relationship union features (num_rels, hidden_dim)
        :param pairwise_auged_obj_feats: the pairwise augmented object features  (num_rels, hidden_dim)
        :param frq_rep: object prediction results for label space co-occurrence prediction
                            (num_rels, ) or  (num_rels, num_classes)
        :param use_label_dist: instead of using label, we can use the probability distribution too.
        :param mean_ctx: for causalities inference
        :return:
        """

        if mean_ctx:
            pairwise_auged_obj_feats = pairwise_auged_obj_feats.mean(-1).unsqueeze(-1)

        if self.union_single_not_match:
            union_feat = self.up_dim_linear(union_feat)

        rel_logits = self.rel_classifier(union_feat + pairwise_auged_obj_feats)

        if self.freq_bias is not None:
            if use_label_dist:
                frq_dists = self.freq_bias.index_with_probability(frq_rep)
            else:
                frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
            union_dists = rel_logits + frq_dists
        else:
            union_dists = rel_logits

        return union_dists

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("RelatednessTestPredictor")
class RelatednessTestPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(RelatednessTestPredictor, self).__init__()
        self.cfg = config
        # todo no attribute recognition now
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        assert self.attribute_on == False

        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.freq_bias = None
        if cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.pairwise_obj_feat_updim_fc = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.output_fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        self.rel_classifier = build_classifier(self.pooling_dim, self.num_rel_cls, bias=True)

        self.rel_pn_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.pooling_dim, self.pooling_dim)
        )
        self.relatedness_model = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        self.pre_clser_type = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD
        self.pre_clser_loss_type = (
            config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRE_CLSER_LOSS
        )

        self.rel_aware_model_pretrain = False
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        if self.rel_aware_model_on:
            self.rel_aware_model_pretrain = False
            self.rel_pn_module = make_relation_confidence_aware_module(self.pooling_dim)

            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.obj_pair_feature_extractor = PairwiseFeatureExtractor(
            config, obj_classes, rel_classes, in_channels
        )

        # initialize layer parameters
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                *[
                    nn.Linear(32, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.pooling_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        layer_init(
            self.pairwise_obj_feat_updim_fc, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True
        )
        self.init_classifier_weight()

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim_linear = nn.Linear(
                config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim
            )
            layer_init(self.up_dim_linear, xavier=True)
        else:
            self.union_single_not_match = False

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.rel_aware_model_pretrain = True

    def end_preclser_relpn_pretrain(self):
        self.rel_aware_model_pretrain = False

    def pair_feature_generate(
        self,
        roi_features,
        proposals,
        rel_pair_idxs,
        num_objs,
        obj_boxs,
        logger,
        ctx_average=False,
    ):
        """

        :param roi_features: object features
        :param proposals:  object proposals
        :param rel_pair_idxs: relation pair matrix
        :param num_objs: the object numbers of each images in batch
        :param obj_boxs: relation pair boxes geometry information list(Tensor(n_pairs, 12))
        :param logger:
        :param ctx_average:
        :return:
            obj_pair_feat4rel_rep:
                use object feat to represent the relationship feature (num_rels, hidden_dim)
            pair_preds_labels, relationship prediction labels (num_rels, )
            pair_bbox_geo_info, the pair boxes relative geometry information, (num_rels, 9)
            pair_obj_probs, the pair objects prediction probability distribution
            rel_binary_preds: relationship affinity prediction results
            obj_dist_prob, object prediction probability distribution
            pairwise_obj_feats_fused
            obj_dist_list
        """
        # encode context information
        (
            obj_dist_logits,
            obj_preds_labels,
            pairwise_obj_feats,
            rel_binary_preds,
        ) = self.obj_pair_feature_extractor(
            roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average
        )

        obj_dist_prob = F.softmax(obj_dist_logits, dim=-1)

        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(pairwise_obj_feats)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
            pairwise_obj_feats_fused.size(0), 2, self.edge_dim
        )
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds_labels = obj_preds_labels.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dist_logits.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_preds_labels = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds_labels, obj_boxs, obj_prob_list
        ):
            obj_pair_feat4rel_rep.append(
                torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
            )
            pair_preds_labels.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
            pair_obj_probs.append(
                torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2)
            )
            pair_bboxs_info.append(
                get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]])
            )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        pair_preds_labels = cat(pair_preds_labels, dim=0)
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)

        obj_pair_feat4rel_rep = self.output_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return (
            obj_pair_feat4rel_rep,
            pair_preds_labels,
            pair_bbox_geo_info,
            pair_obj_probs,
            rel_binary_preds,
            obj_dist_prob,
            obj_dist_list,
        )

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param proposals: object [BoxList]
        :param rel_pair_idxs: the object pairing matrix list(Tensor)
        :param rel_labels:  GT rel_labels for loss computation list(Tensor)
        :param rel_binarys: GT rel_binary labels for loss computation list(Tensor)
        :param roi_features: object features list(Tensor)
        :param union_features: the union box features of each relationships list((Tensor)) or
                               list((tuple(Tensor))) contains visual features and  spatial box masks features
        :param logger:
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        (
            obj_pair_feat4rel_rep,
            pair_preds_labels,
            pair_bbox_geo_info,
            pair_obj_probs,
            rel_binary_preds,
            obj_dist_prob,
            obj_dist_list,
        ) = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger
        )

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * spatial_conv_feats

        if self.spatial_for_vision:
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)

        #############################
        # relness module
        rel_cnf_logits = None
        if self.rel_pn_module is not None:
            if self.union_single_not_match:

                rel_pn_feat = self.rel_pn_fc(
                    self.up_dim_linear(union_features) + obj_pair_feat4rel_rep
                )
            else:
                rel_pn_feat = self.rel_pn_fc(union_features + obj_pair_feat4rel_rep)

            rel_cnf_logits, relatedness_scores = self.rel_pn_module(
                rel_pn_feat, proposals, rel_pair_idxs
            )

            # naive rel pn module: use the detection product
            ######################################
            rel_prop_pairs_relness_batch = []
            for idx in range(len(relatedness_scores)):
                rel_ind_i = rel_pair_idxs[idx]
                prop = proposals[idx]
                related_matrix = relatedness_scores[idx]

                det_score = prop.get_field("pred_scores")
                related_matrix[rel_ind_i[:, 0], rel_ind_i[:, 1]] = (
                    det_score[rel_ind_i[:, 0]] * det_score[rel_ind_i[:, 1]]
                )

                rel_prop_pairs_relness_batch.append(related_matrix)

            relatedness_scores = rel_prop_pairs_relness_batch

            ######################################
            for idx, prop in enumerate(proposals):
                prop.add_field("relness_mat", relatedness_scores[idx].unsqueeze(-1))
        #############################

        # todo fusion calculate not TDE split style
        rel_dists = self.calculate_logits(
            union_features, obj_pair_feat4rel_rep, pair_preds_labels, use_label_dist=False
        )
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        if rel_cnf_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                loss_rel_pre_cls = None
            else:
                loss_rel_pre_cls = self.rel_aware_loss_eval(rel_cnf_logits, rel_labels)

            add_losses[f"pre_rel_classify_loss_iter"] = loss_rel_pre_cls

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(
                0
            ).view(-1)
        return holder

    def calculate_logits(
        self,
        union_feat,
        pairwise_auged_obj_feats,
        frq_rep,
        use_label_dist=True,
        mean_ctx=False,
    ):
        """

        :param union_feat: relationship union features (num_rels, hidden_dim)
        :param pairwise_auged_obj_feats: the pairwise augmented object features  (num_rels, hidden_dim)
        :param frq_rep: object prediction results for label space co-occurrence prediction
                            (num_rels, ) or  (num_rels, num_classes)
        :param use_label_dist: instead of using label, we can use the probability distribution too.
        :param mean_ctx: for causalities inference
        :return:
        """

        if mean_ctx:
            pairwise_auged_obj_feats = pairwise_auged_obj_feats.mean(-1).unsqueeze(-1)

        if self.union_single_not_match:
            union_feat = self.up_dim_linear(union_feat)

        rel_logits = self.rel_classifier(union_feat + pairwise_auged_obj_feats)

        if self.freq_bias is not None:
            if use_label_dist:
                frq_dists = self.freq_bias.index_with_probability(frq_rep)
            else:
                frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
            union_dists = rel_logits + frq_dists
        else:
            union_dists = rel_logits

        return union_dists

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("KERNPredictor")
class KERNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(KERNPredictor, self).__init__()
        self.cfg = config
        # todo no attribute recognition now
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        assert self.attribute_on == False

        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.freq_bias = None
        if cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.input_dim = (
            config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        )  # from det ROI_BOX_HEAD extractor
        self.hidden_dim = (
            config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        )  # the dim that inside module use
        self.pooling_dim = (
            config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        )  # output dimension

        # post decoding
        self.edge_dim = self.hidden_dim
        self.pairwise_obj_feat_updim_fc = None
        if config.MODEL.ROI_RELATION_HEAD.KERN_MODULE.FUSE_PAIRWISE_OBJ_FEATURES:
            self.pairwise_obj_feat_updim_fc = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            layer_init(
                self.pairwise_obj_feat_updim_fc,
                10.0 * (1.0 / self.hidden_dim) ** 0.5,
                normal=True,
            )

        self.output_fc = nn.Sequential(
            make_fc(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )

        self.rel_classifier = build_classifier(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        self.init_classifier_weight()

        if self.pooling_dim != self.input_dim:
            self.union_single_not_match = True
            self.rel_feat_updim_fc = nn.Linear(self.pooling_dim, self.input_dim)
            layer_init(self.rel_feat_updim_fc, xavier=True)
            self.rel_feat_downdim_fc = nn.Linear(self.input_dim, self.pooling_dim)
            layer_init(self.rel_feat_downdim_fc, xavier=True)
        else:
            self.union_single_not_match = False

        self.fuse_pairwise_obj_features = (
            cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.FUSE_PAIRWISE_OBJ_FEATURES
        )

        self.obj_pair_feature_extractor = InstanceFeaturesAugments(
            config, obj_classes, self.input_dim, self.fuse_pairwise_obj_features
        )

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                *[
                    nn.Linear(32, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.pooling_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.KERN_rel_reasoning = GGNNRelReason(
            self.num_obj_cls,
            self.num_rel_cls,
            inst_feat_dim=self.pooling_dim,
            rel_feat_dim=self.pooling_dim,
            output_dim=self.pooling_dim,
            hidden_dim=cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.GRAPH_HIDDEN_DIM,
            time_step_num=cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.MESSAGE_PASSING_STEP,
            use_knowledge=cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.STATISTICS_PRIOR_KNOWLEDGE,
            knowledge_matrix=cfg.MODEL.ROI_RELATION_HEAD.KERN_MODULE.REL_PRIOR_MATRIX_DIR,
        )

        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_union_feat", torch.zeros(self.pooling_dim))

        # self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()

    def pair_feature_generate(
        self,
        roi_features,
        proposals,
        rel_pair_idxs,
        num_objs,
        obj_boxs,
        logger,
        ctx_average=False,
    ):
        """

        :param roi_features: object features
        :param proposals:  object proposals
        :param rel_pair_idxs: relation pair matrix
        :param num_objs: the object numbers of each images in batch
        :param obj_boxs: relation pair boxes geometry information list(Tensor(n_pairs, 12))
        :param logger:
        :param ctx_average:
        :return:
            obj_pair_feat4rel_rep:
                use object feat to represent the relationship feature (num_rels, hidden_dim)
            pair_preds_labels, relationship prediction labels (num_rels, )
            pair_bbox_geo_info, the pair boxes relative geometry information, (num_rels, 9)
            pair_obj_probs, the pair objects prediction probability distribution
            rel_binary_preds: relationship affinity prediction results
            obj_dist_prob, object prediction probability distribution
            pairwise_obj_feats_fused
            obj_dist_list
        """

        (
            obj_representation4rel,
            augment_obj_feat,
            obj_dist_logits,
            obj_preds_labels,
        ) = self.obj_pair_feature_extractor(
            roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average
        )

        obj_dist_prob = F.softmax(obj_dist_logits, dim=-1)

        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        head_reps = [None for _ in range(len(num_objs))]
        tail_reps = [None for _ in range(len(num_objs))]
        if obj_representation4rel is not None:
            pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(obj_representation4rel)
            pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(
                pairwise_obj_feats_fused.size(0), 2, self.edge_dim
            )
            head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.edge_dim)
            tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.edge_dim)
            # split
            head_reps = head_rep.split(num_objs, dim=0)
            tail_reps = tail_rep.split(num_objs, dim=0)

        obj_preds_labels = obj_preds_labels.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dist_logits.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_preds_labels = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds_labels, obj_boxs, obj_prob_list
        ):
            if obj_representation4rel is not None:
                obj_pair_feat4rel_rep.append(
                    torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
                )
            pair_preds_labels.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
            pair_obj_probs.append(
                torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2)
            )
            pair_bboxs_info.append(
                get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]])
            )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)
        pair_preds_labels = cat(pair_preds_labels, dim=0)
        if self.fuse_pairwise_obj_features:
            obj_pair_feat4rel_rep = cat(
                obj_pair_feat4rel_rep, dim=0
            )  # (num_rel, hidden_dim * 2)
            obj_pair_feat4rel_rep = self.output_fc(
                obj_pair_feat4rel_rep
            )  # (num_rel, hidden_dim)
        else:
            obj_pair_feat4rel_rep = None

        return (
            augment_obj_feat,
            obj_pair_feat4rel_rep,
            pair_bbox_geo_info,
            pair_preds_labels,
            pair_obj_probs,
            obj_dist_prob,
            obj_dist_list,
        )

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param proposals: object [BoxList]
        :param rel_pair_idxs: the object pairing matrix list(Tensor)
        :param rel_labels:  GT rel_labels for loss computation list(Tensor)
        :param rel_binarys: GT rel_binary labels for loss computation list(Tensor)
        :param roi_features: object features list(Tensor)
        :param union_features: the union box features of each relationships list((Tensor)) or
                               list((tuple(Tensor))) contains visual features and  spatial box masks features
        :param logger:
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        (
            augment_obj_feat,
            obj_pair_feat4rel_rep,
            pair_bbox_geo_info,
            pair_preds_labels,
            pair_obj_probs,
            obj_dist_prob,
            obj_dist_list,
        ) = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger
        )
        # todo: GNN on objects features

        # generate the avg blank features for causalities comparison
        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                (
                    avg_post_ctx_rep,
                    _,
                    _,
                    avg_pair_obj_prob,
                    _,
                    _,
                    _,
                ) = self.pair_feature_generate(
                    roi_features,
                    proposals,
                    rel_pair_idxs,
                    num_objs,
                    obj_boxs,
                    logger,
                    ctx_average=True,
                )
        # GNN on relationship features
        if self.fuse_pairwise_obj_features:
            if self.separate_spatial:
                union_features, spatial_conv_feats = union_features
                obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * spatial_conv_feats

            if self.spatial_for_vision:
                obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(
                    pair_bbox_geo_info
                )

            if self.union_single_not_match:
                rel_reasoning_features = obj_pair_feat4rel_rep + self.rel_feat_updim_fc(
                    union_features
                )
            else:
                rel_reasoning_features = obj_pair_feat4rel_rep + union_features
        else:
            rel_reasoning_features = union_features
        if self.union_single_not_match:
            rel_reasoning_features = self.rel_feat_downdim_fc(rel_reasoning_features)
        rel_reasonion_out_feats = self.KERN_rel_reasoning(
            augment_obj_feat, rel_reasoning_features, pair_preds_labels, rel_pair_idxs
        )

        rel_dists = self.calculate_logits(
            rel_reasonion_out_feats, pair_preds_labels, use_label_dist=False
        )

        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.effect_analysis:
            # todo this part is un-developed
            if self.training:
                # todo currently no auxiliary loss
                # untreated average feature
                if self.spatial_for_vision:
                    self.untreated_spt = self.moving_average(
                        self.untreated_spt, pair_bbox_geo_info
                    )
                if self.separate_spatial:
                    self.untreated_conv_spt = self.moving_average(
                        self.untreated_conv_spt, spatial_conv_feats
                    )
                self.avg_post_ctx = self.moving_average(
                    self.avg_post_ctx, obj_pair_feat4rel_rep
                )
                self.untreated_union_feat = self.moving_average(
                    self.untreated_union_feat, union_features
                )

            else:
                with torch.no_grad():
                    # untreated spatial
                    if self.spatial_for_vision:
                        avg_spt_rep = self.spt_emb(
                            self.untreated_spt.clone().detach().view(1, -1)
                        )
                    # untreated context
                    avg_ctx_rep = (
                        avg_post_ctx_rep * avg_spt_rep
                        if self.spatial_for_vision
                        else avg_post_ctx_rep
                    )
                    avg_ctx_rep = (
                        avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1)
                        if self.separate_spatial
                        else avg_ctx_rep
                    )
                    # untreated visual
                    avg_vis_rep = self.untreated_union_feat.clone().detach().view(1, -1)
                    # untreated category dist
                    avg_frq_rep = avg_pair_obj_prob

                if self.effect_type == "TDE":  # TDE of CTX
                    rel_dists = self.calculate_logits(
                        union_features, obj_pair_feat4rel_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
                elif self.effect_type == "NIE":  # NIE of FRQ
                    rel_dists = self.calculate_logits(
                        union_features, avg_ctx_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                elif self.effect_type == "TE":  # Total Effect
                    rel_dists = self.calculate_logits(
                        union_features, obj_pair_feat4rel_rep, pair_obj_probs
                    ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
                else:
                    assert self.effect_type == "none"
                    pass
                rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(
                0
            ).view(-1)
        return holder

    def calculate_logits(
        self,
        rel_feats,
        frq_rep,
        use_label_dist=True,
    ):
        """

        :param rel_feats: relationship union features (num_rels, hidden_dim)
        :param frq_rep: object prediction results for label space co-occurrence prediction
                            (num_rels, ) or  (num_rels, num_classes)
        :param use_label_dist: instead of using label, we can use the probability distribution too.
        :return:
        """

        rel_logits = self.rel_classifier(rel_feats)

        if self.freq_bias is not None:
            if use_label_dist:
                frq_dists = self.freq_bias.index_with_probability(frq_rep)
            else:
                frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
            union_dists = rel_logits + frq_dists
        else:
            union_dists = rel_logits

        return union_dists

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(
                config, obj_classes, rel_classes, statistics, in_channels
            )
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print("ERROR: Invalid Context Layer")
            assert False

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)

            self.ctx_compress = build_classifier(
                self.pooling_dim, self.num_rel_cls, bias=False
            )
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(
                *[
                    nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            layer_init(self.post_cat[0], xavier=True)
            self.ctx_compress = build_classifier(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = build_classifier(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == "gate":
            self.ctx_gate_fc = build_classifier(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        self.init_classifier_weight()

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                *[
                    nn.Linear(32, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.pooling_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.auxiliary_loss_on = config.MODEL.ROI_RELATION_HEAD.CAUSAL.AUXILIARY_LOSS

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def init_classifier_weight(self):

        if cfg.MODEL.ROI_RELATION_HEAD.CLASSIFIER == "linear":
            if not self.use_vtranse:
                layer_init(self.ctx_compress, xavier=True)
            layer_init(self.vis_compress, xavier=True)

        elif cfg.MODEL.ROI_RELATION_HEAD.CLASSIFIER in ["weighted_norm", "cosine_similarity"]:
            self.vis_compress.reset_parameters()
            self.ctx_compress.reset_parameters()

    def pair_feature_generate(
        self,
        roi_features,
        proposals,
        rel_pair_idxs,
        num_objs,
        obj_boxs,
        logger,
        ctx_average=False,
    ):
        """

        :param roi_features: object features
        :param proposals:  object proposals
        :param rel_pair_idxs: relation pair matrix
        :param num_objs: the object numbers of each images in batch
        :param obj_boxs: relation pair boxes geometry information list(Tensor(n_pairs, 12))
        :param logger:
        :param ctx_average:
        :return:
        """
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(
            roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average
        )
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(
            edge_ctx
        )  # divide to head and tail representation of relationships
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(
            rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list
        ):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(
                    torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1)
                )
            pair_preds.append(
                torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
            )
            pair_obj_probs.append(
                torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2)
            )
            pair_bboxs_info.append(
                get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]])
            )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox_geo_feat = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return (
            post_ctx_rep,
            pair_pred,
            pair_bbox_geo_feat,
            pair_obj_probs,
            binary_preds,
            obj_dist_prob,
            edge_rep,
            obj_dist_list,
        )

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        """

        :param proposals: object [BoxList]
        :param rel_pair_idxs: the object pairing matrix list(Tensor)
        :param rel_labels:  GT rel_labels for loss computation list(Tensor)
        :param rel_binarys: GT rel_binary labels for loss computation list(Tensor)
        :param roi_features: object features list(Tensor)
        :param union_features: the union box features of each relationships list(Tensor)
        :param logger:
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        (
            post_ctx_rep,
            pair_pred,
            pair_bbox,
            pair_obj_probs,
            binary_preds,
            obj_dist_prob,
            edge_rep,
            obj_dist_list,
        ) = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger
        )

        # generate the avg blank features for causalities comparison
        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                (
                    avg_post_ctx_rep,
                    _,
                    _,
                    avg_pair_obj_prob,
                    _,
                    _,
                    _,
                    _,
                ) = self.pair_feature_generate(
                    roi_features,
                    proposals,
                    rel_pair_idxs,
                    num_objs,
                    obj_boxs,
                    logger,
                    ctx_average=True,
                )

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(
            union_features, post_ctx_rep, pair_pred, use_label_dist=False
        )
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            # todo: but why?
            if self.auxiliary_loss_on:
                obj_ctx_rel_logits = self.ctx_compress(post_ctx_rep)
                add_losses["auxiliary_ctx"] = F.cross_entropy(
                    obj_ctx_rel_logits[rel_labels != -1], rel_labels[rel_labels != -1]
                )
                if not (self.fusion_type == "gate"):
                    union_feat_rel_logit = self.vis_compress(union_features)
                    add_losses["auxiliary_vis"] = F.cross_entropy(
                        union_feat_rel_logit[rel_labels != -1], rel_labels[rel_labels != -1]
                    )
                    obj_pair_freq_bias_logit = self.freq_bias.index_with_labels(
                        pair_pred.long()
                    )
                    add_losses["auxiliary_frq"] = F.cross_entropy(
                        obj_pair_freq_bias_logit[rel_labels != -1],
                        rel_labels[rel_labels != -1],
                    )

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(
                    self.untreated_conv_spt, spatial_conv_feats
                )
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = (
                    avg_post_ctx_rep * avg_spt_rep
                    if self.spatial_for_vision
                    else avg_post_ctx_rep
                )
                avg_ctx_rep = (
                    avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1)
                    if self.separate_spatial
                    else avg_ctx_rep
                )
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == "TDE":  # TDE of CTX
                rel_dists = self.calculate_logits(
                    union_features, post_ctx_rep, pair_obj_probs
                ) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == "NIE":  # NIE of FRQ
                rel_dists = self.calculate_logits(
                    union_features, avg_ctx_rep, pair_obj_probs
                ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == "TE":  # Total Effect
                rel_dists = self.calculate_logits(
                    union_features, post_ctx_rep, pair_obj_probs
                ) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == "none"
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(
                0
            ).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)

        if self.fusion_type != "features":
            vis_dists = self.vis_compress(vis_rep)
            ctx_dists = self.ctx_compress(ctx_rep)

            if self.fusion_type == "gate":
                ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
                union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
                # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
                # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
                # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
                # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
                # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
                # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
                # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

            elif self.fusion_type == "sum":
                if cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.OBJ_PAIR_LABEL_FREQUENCY_BIAS_BRANCH:
                    union_dists = vis_dists + ctx_dists + frq_dists
                else:
                    union_dists = vis_dists + ctx_dists
            else:
                raise ValueError("invalid fusion type")

        elif self.fusion_type == "features":
            union_dists = self.vis_compress(vis_rep + ctx_rep)
        else:
            raise ValueError("invalid fusion type")

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
