"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

from PIL import Image
from collections import OrderedDict

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

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
        self.gres_tool = GRES_Inference()

    def __getitem__(self, index):
        ann = self.annotation[index]

        question = self.text_processor(ann["question"])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        raw_image = np.array(image)

        focus_dict = {}
        focus_dict['image'] = raw_image
        focus_dict['text_input'] = question

        focus_dict = self.gres_tool.gres_data2feature(focus_dict)

        # resize_image, output = self.gres_model.infer(focus_dict)
        # raw_focus_image = resize_image * output['mask'][0].cpu()
        # focus_image = np.array(raw_focus_image.permute(1, 2, 0))
        # focus_image = Image.fromarray(np.uint8(focus_image))
        # focus_image = self.vis_processor(focus_image)

        image = self.vis_processor(image)

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        # answers = list(answer_weight.keys())
        # weights = list(answer_weight.values())
        best_answer = max(answer_weight, key=answer_weight.get)

        return {
            "image": image,
            "focus_dict": focus_dict,
            "text_input": question,
            "text_output": best_answer,

        }

    def collater(self, samples):
        image_list, focus_dict_list, question_list, answer_list = [], [], [], [],

        for sample in samples:
            image_list.append(sample["image"])

            focus_dict_list.append(sample["focus_dict"])

            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)

        return {
            "image": torch.stack(image_list, dim=0),
            "focus_dict": focus_dict_list,
            "text_input": question_list,
            "text_output": answer_list,
        }


class COCOVQAGRESEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        # Data sampling
        self.annotation = self.annotation[:100000]

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

        # self.gres_model = GRES_Inference()
        self.gres_tool = GRES_Inference()

    def __getitem__(self, index):
        ann = self.annotation[index]

        question = self.text_processor(ann["question"])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        raw_image = np.array(image)

        focus_dict = {}
        focus_dict['image'] = raw_image
        focus_dict['text_input'] = question

        focus_dict = self.gres_tool.gres_data2feature(focus_dict)

        # resize_image, output = self.gres_model.infer(focus_dict)
        # raw_focus_image = resize_image * output['mask'][0].cpu()
        # focus_image = np.array(raw_focus_image.permute(1, 2, 0))
        # focus_image = Image.fromarray(np.uint8(focus_image))
        # focus_image = self.vis_processor(focus_image)

        image = self.vis_processor(image)

        return {
            "image": image,
            "focus_dict": focus_dict,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

    def collater(self, samples):
        image_list, focus_dict_list, text_input_list, question_id_list, instance_id_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])

            focus_dict_list.append(sample["focus_dict"])

            text_input_list.append(sample["text_input"])

            question_id_list.append(sample["question_id"])

            instance_id_list.append(sample["instance_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "focus_dict": focus_dict_list,
            "text_input": text_input_list,
            "question_id": torch.tensor(question_id_list),
            "instance_id": instance_id_list,
        }