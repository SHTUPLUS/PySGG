import argparse
import json
import os
from collections import OrderedDict

import torch
from PIL import Image

from thesis_demo import my_tranform as MT
from pysgg.config import cfg 
from pysgg.modeling.detector import build_detection_model
from pysgg.structures.image_list import to_image_list
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.logger import setup_logger

setup_logger("pysgg", None, 0)




class VRDPredictor:
    def __init__(self, cfg_dir):

        cfg.merge_from_file("/root/projects/Scene-Graph-Benchmark.pytorch/demo/vis_demo/thesis_demo/cfgs/config.yml")
        cfg.freeze()

        self.det_thres = 0.6
        self.rel_thres = 0.03

        

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device("cpu")
        self.transforms = self._build_transform()
        self._init_category_dict()

        self.model = build_detection_model(cfg)
        self.model.to(cfg.MODEL.DEVICE)
        self.model.eval()
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="thesis_demo/cache")
        checkpoint, ckpt_fname = checkpointer.load(cfg.MODEL.WEIGHT)
        print("load model %s" % ckpt_fname)

    def _init_category_dict(self):

        def load_info(dict_file, add_bg=True):
            """
            Loads the file containing the visual genome label meanings
            """
            info = json.load(open(dict_file, 'r'))
            if add_bg:
                info['label_to_idx']['__background__'] = 0
                info['predicate_to_idx']['__background__'] = 0

            class_to_ind = info['label_to_idx']
            predicate_to_ind = info['predicate_to_idx']
            ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
            ind_to_predicates = sorted(
                predicate_to_ind, key=lambda k: predicate_to_ind[k])

            return ind_to_classes, ind_to_predicates

        self.obj_cls_list, self.rel_cls_list  = load_info("/root/projects/Scene-Graph-Benchmark.pytorch/datasets/vg/stanford_spilt/stanford_spilt/VG-SGG-dicts.json")
        # categories utilities building
        self.obj_cls_num = len(self.obj_cls_list)
        self.rel_cls_num = len(self.rel_cls_list)

    def _build_transform(self):
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST

        to_bgr255 = cfg.INPUT.TO_BGR255
        normalize_transform = MT.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )

        transform = MT.Compose(
            [
                MT.Resize(min_size, max_size),
                MT.ToTensor(),
                normalize_transform,
            ]
        )
        return transform

    def inference(self, src_image):
        """

        :param src_image:
        :return:
        """
        img = Image.open(src_image).convert('RGB')

        init_img_size = tuple(img.size)

        img = self.transforms(img)
        image_list = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        with torch.no_grad():
            results = self.model(image_list)

        # collect and move result to cpu memory
        if results is not None:
            moved_res = [o.to(self.cpu_device) for o in results]
            results = moved_res
        to_show_result = self.interpret_result(results[0], init_img_size, src_image)
        return to_show_result

    def interpret_result(self, rel_res, init_img_size, img_dir):
        # filter the rel_res
        rel_res = rel_res.resize(init_img_size).convert("xywh")

        to_show_res = []

        inst_labels = rel_res.get_field("pred_labels")
        inst_score = rel_res.get_field("pred_scores")

        pred_rel_scores = rel_res.get_field("pred_rel_scores")
        rel_scores, rel_class = pred_rel_scores[:, 1:].max(dim=1)
        rel_class += 1

        keep_idx = 500


        selected_rel_pair_idx = rel_res.get_field("rel_pair_idxs")[:keep_idx]
        rel_class = rel_class[:keep_idx]
        rel_scores = rel_scores[:keep_idx]

        print(selected_rel_pair_idx)

        for idx, each_pair in enumerate(selected_rel_pair_idx):
            subj_idx = each_pair[0].item()
            obj_idx = each_pair[1].item()

            subj_score = inst_score[subj_idx]
            obj_score = inst_score[obj_idx]
            # if not subj_score > self.det_thres and not obj_score > self.det_thres:
            #     continue

            subj_inst_box = rel_res.bbox[subj_idx]
            sub_label_id = inst_labels[subj_idx]

            obj_inst_box = rel_res.bbox[obj_idx]
            obj_label_id = inst_labels[obj_idx]

            phra_label_id = rel_class[idx].item()
            phra_score = rel_scores[idx].item()

            trp_score = phra_score * obj_score * subj_score
            trp_score = trp_score.item()
            # if trp_score < 0.15:
            #     continue
            to_show_res.append({
                "text": [self.obj_cls_list[sub_label_id],
                         self.rel_cls_list[phra_label_id],
                         self.obj_cls_list[obj_label_id]],
                "bboxes": [
                    box_to_dict(build_box(subj_inst_box, init_img_size), "s"),
                    box_to_dict(build_box(obj_inst_box, init_img_size), "o")
                ],
                "score": trp_score

            })


        to_show_res.sort(key=lambda a: a['score'], reverse=True,)

        for each in to_show_res:
            print(each['text'],each['score'])

        img_url = os.path.join("http://10.15.89.41:32783/static/img_cache",
                               img_dir.split('/')[-1])
        return {
            "relationships": to_show_res,
            "image_width": init_img_size[0],
            "url":img_url,
            "image_id": 111,
        }


def pack_to_json_str(det_res):
    res_json = json.dumps([det_res])
    res_json = res_json.replace("\"", "&quot;")
    res_json = res_json.replace("\'", "&quot;")
    return res_json


def build_box(box_tensor, init_img_size):

    xywh = [abs(int(each.item())) for each in box_tensor]
    xywh[2] = xywh[2] - 15 if abs(init_img_size[0] - xywh[2]) < 5 else xywh[2]
    xywh[3] = xywh[3] - 15 if abs(init_img_size[1] - xywh[3]) < 5 else xywh[3]
    return xywh


def box_to_dict(box_list, pos):
    if pos == 'o':
        color = "#86DADE"
    else:
        color = "#FF00FF"
    return {
        'x': box_list[0],
        'y': box_list[1],
        'color': color,
        'w': box_list[2],
        'h': box_list[3]
    }




def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    predtor = VRDPredictor("")
    det_res = predtor.inference(img_dir)
    s = pack_to_json_str(det_res)
    with open("out.txt", 'w') as f:
        f.write(s)


