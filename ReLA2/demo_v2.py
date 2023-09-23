#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import pickle
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from detectron2.evaluation import inference_context
from ReLA2.model_instance import get_model, raw_data2feature
from detectron2.engine import default_argument_parser

from ReLA2.Inference import GRES_Inference

gres_model = GRES_Inference()

image_path = '/mnt/local/wwx/LLM_Data/tmp_test/test_01.png'
text_input = 'Traffic light status'


image = Image.open(image_path).convert("RGB")
image = np.array(image)

input_dic = {
        'image': image,
        'text_input': text_input,
    }

with torch.no_grad():
    resize_image, focus_dict = gres_model(input_dic)

    raw_focus_image = resize_image * focus_dict['mask'][0].cpu()
    focus_image = np.array(raw_focus_image.permute(1, 2, 0))
    print("123")

    # resize_image, focus_dict = GRES_model.infer(feature_dic)
    # for res, masker in zip(outputs['images'], outputs['mask']):
    #     res = res * masker
    #     fig = plt.figure()
    #     plt.tight_layout()
    #     plt.imshow(res.to("cpu").permute(1, 2, 0).detach().numpy())
    #     plt.title(f"The image is multiplied by target_masks. target: {input_dic['text_input']}")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()
