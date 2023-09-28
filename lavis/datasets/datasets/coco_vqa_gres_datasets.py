"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import numpy as np

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict

from ReLA2.Inference import GRES_Inference


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class COCOVQAGRESDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.gres_model = GRES_Inference()

    def __getitem__(self, index):
        ann = self.annotation[index]

        question = self.text_processor(ann["question"])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        raw_image = np.array(image)

        focus_dict = {}
        focus_dict['image'] = raw_image
        focus_dict['text_input'] = question[0]
        resize_image, focus_dict = self.gres_model.infer(focus_dict)

        raw_focus_image = resize_image * focus_dict['mask'][0].cpu()
        focus_image = np.array(raw_focus_image.permute(1, 2, 0))

        focus_image = Image.fromarray(np.uint8(focus_image))
        focus_image = self.vis_processor(focus_image)

        image = self.vis_processor(image)

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "focus_image": focus_image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class COCOVQAGRESEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        # select top 100
        self.annotation = self.annotation[:500]

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        self.gres_model = GRES_Inference()

    def __getitem__(self, index):
        ann = self.annotation[index]

        question = self.text_processor(ann["question"])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        raw_image = np.array(image)

        focus_dict = {}
        focus_dict['image'] = raw_image
        focus_dict['text_input'] = question[0]
        resize_image, focus_dict = self.gres_model.infer(focus_dict)

        raw_focus_image = resize_image * focus_dict['mask'][0].cpu()
        focus_image = np.array(raw_focus_image.permute(1, 2, 0))

        focus_image = Image.fromarray(np.uint8(focus_image))
        focus_image = self.vis_processor(focus_image)

        image = self.vis_processor(image)

        return {
            "image": image,
            "focus_image": focus_image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
