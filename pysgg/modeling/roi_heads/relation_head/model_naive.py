import torch
from torch import nn
from torch.nn import functional as F

from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.classifier import DotProductClassifier
from pysgg.modeling.roi_heads.relation_head.utils_relation import (
    obj_prediction_nms,
)
from pysgg.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot


# simlar to the Motifs LSTM pipeline, but we just use to generate the object pair features
# such as labels space embedding


class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.word_embed_feats_on = self.cfg.MODEL.ROI_RELATION_HEAD.WORD_EMBEDDING_FEATURES
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(
                self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim
            )
            self.obj_embed_on_1stg_pred = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_2stg_pred = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_1stg_pred.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_2stg_pred.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(
            *[
                nn.Linear(9, 32),
                nn.BatchNorm1d(32, momentum=0.001),
                nn.Linear(32, self.geometry_feat_dim),
                nn.ReLU(inplace=True),
            ]
        )
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_feat_refine_hidden_fc = make_fc(
            self.obj_dim + self.embed_dim + self.geometry_feat_dim, self.hidden_dim
        )
        self.edges_refine_hidden_fc = make_fc(
            self.hidden_dim + self.obj_dim + self.embed_dim, self.hidden_dim
        )
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        self.obj_reclassify_on_auged_feats = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        )
        self.rel_obj_mulit_task = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        self.obj_classifier = None
        if self.obj_reclassify_on_auged_feats or self.rel_obj_mulit_task:
            self.obj_classifier = DotProductClassifier(self.hidden_dim, self.num_obj_classes)
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        if self.effect_analysis:
            self.register_buffer(
                "untreated_obj_pairwise_dowdim_feat", torch.zeros(self.hidden_dim)
            )
            self.register_buffer(
                "untreated_obj_init_feat",
                torch.zeros(self.obj_dim + self.embed_dim + self.geometry_feat_dim),
            )
            self.register_buffer(
                "untreated_obj_pairwised_feat", torch.zeros(self.obj_dim + self.embed_dim)
            )

    def object_feature_refine_reclassify(
        self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False
    ):
        """
        Object feature refinement by embedding representation and redo classification on new representation.
        all vectors from each images of batch are cat together
        :param obj_feats: [num_obj, ROI feat dim + object embedding0 dim + geometry_feat_dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param proposals: BoxList for objects
        :param boxes_per_cls: regressed boxes for each categories

        :return: obj_pred_logits: [num_obj, #classes] new probability distribution.
                 obj_preds: [num_obj, ] argmax of that distribution.
                 augmented_obj_features: [num_obj, #feats] For later!
        """

        # fuse the ebedding featuresrefined_obj_logits
        augmented_obj_features = self.obj_feat_refine_hidden_fc(obj_feats)  # map to hidden_dim

        # untreated decoder input
        batch_size = augmented_obj_features.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            augmented_obj_features = self.untreated_obj_pairwise_dowdim_feat.view(
                1, -1
            ).expand(batch_size, -1)

        if self.training and self.effect_analysis:
            self.untreated_obj_pairwise_dowdim_feat = self.moving_average(
                self.untreated_obj_pairwise_dowdim_feat, augmented_obj_features
            )

        # reclassify on the fused object features
        # Decode in order
        if self.mode != "predcls":

            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in proposals], dim=0
            )
            obj_pred_logits = cat(
                [each_prop.get_field("predict_logits") for each_prop in proposals], dim=0
            )
            if self.rel_obj_mulit_task:
                refined_obj_logits = self.obj_classifier(augmented_obj_features)
                # here we use the logits refinements by adding
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits += refined_obj_logits
                # the logits refinements by replace
                elif self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                # update the pred_labels
                assert boxes_per_cls is not None
                refined_obj_pred_labels = obj_prediction_nms(
                    boxes_per_cls, obj_pred_logits, nms_thresh=0.5
                )
                obj_pred_labels = refined_obj_pred_labels
        else:
            assert obj_labels is not None
            obj_pred_labels = obj_labels
            obj_pred_logits = to_onehot(obj_pred_labels, self.num_obj_classes)

        return augmented_obj_features, obj_pred_labels, obj_pred_logits

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(
                0
            ).view(-1)
        return holder

    def forward(
        self,
        inst_roi_feats,
        inst_proposals,
        rel_pair_idxs,
        logger=None,
        all_average=False,
        ctx_average=False,
    ):
        """

        :param inst_roi_feats: instance ROI features(batch cancate), Tensor
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :param logger:
        :param all_average:
        :param ctx_average:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                obj_embed_by_pred_dist = self.obj_embed_on_1stg_pred(obj_labels.long())
            else:
                obj_logits = cat(
                    [proposal.get_field("predict_logits") for proposal in inst_proposals],
                    dim=0,
                ).detach()
                obj_embed_by_pred_dist = (
                    F.softmax(obj_logits, dim=1) @ self.obj_embed_on_1stg_pred.weight
                )

        # box positive geometry embedding
        assert inst_proposals[0].mode == "xyxy"
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        batch_size = inst_roi_feats.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_init_feat.view(1, -1).expand(batch_size, -1)
        else:
            if self.word_embed_feats_on:
                obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
            else:
                obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)

        boxes_per_cls = None
        if self.mode in ["sgdet", "sgcls"]:
            boxes_per_cls = cat(
                [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
            )  # comes from post process of box_head
        # object level contextual feature
        (
            augment_obj_feat,
            obj_pred_labels,
            obj_pred_logits,
        ) = self.object_feature_refine_reclassify(
            obj_pre_rep, inst_proposals, obj_labels, boxes_per_cls, ctx_average=ctx_average
        )
        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:
            obj_embed_by_pred_labels = self.obj_embed_on_2stg_pred(obj_pred_labels.long())

        # average action in test phrase for causal effect analysis
        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            # average the embedding and initial ROI features
            obj_representation4rel = cat(
                (
                    self.untreated_obj_pairwised_feat.view(1, -1).expand(batch_size, -1),
                    augment_obj_feat,
                ),
                dim=-1,
            )
        else:
            if self.word_embed_feats_on:
                obj_representation4rel = cat(
                    (obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1
                )
            else:
                obj_representation4rel = cat((inst_roi_feats, augment_obj_feat), -1)

        # mapping to hidden
        obj_representation4rel = self.edges_refine_hidden_fc(obj_representation4rel)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_init_feat = self.moving_average(
                self.untreated_obj_init_feat, obj_pre_rep
            )
            if self.word_embed_feats_on:
                self.untreated_obj_pairwised_feat = self.moving_average(
                    self.untreated_obj_pairwised_feat,
                    cat((obj_embed_by_pred_labels, inst_roi_feats), -1),
                )
            else:
                self.untreated_obj_pairwised_feat = self.moving_average(
                    self.untreated_obj_pairwised_feat, inst_roi_feats
                )
        return obj_pred_logits, obj_pred_labels, obj_representation4rel, None
