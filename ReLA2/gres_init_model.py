#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from detectron2.evaluation import inference_context
from detectron2.engine import default_argument_parser
from ReLA2.model_instance import get_model, raw_data2feature

# TODO: image needs to be source image
class GRES_Model:
    def __init__(self):
        artificial_args = """--config-file
        configs/referring_swin_base.yaml
        --num-gpus
        1
        --dist-url
        auto
        MODEL.WEIGHTS
        /mnt/local/wwx/ckpts/GRES/gres_swin_base.pth
        OUTPUT_DIR
        /mnt/local/wwx/Output/GRES/0918"""

        sys.argv = []
        sys.argv.append('/home/tsingqguo/wwx/paper_worksapce/LAVIS/ReLA2/demo.py')
        for argument in artificial_args.split('\n'):
            sys.argv.append(argument.lstrip())

        args, unknown = default_argument_parser().parse_known_args()
        args.config_file = 'ReLA2/' + args.config_file

        self.GRES_model, self.cfg = get_model(args)


    def infer(self, samples):
        feature_dic = raw_data2feature(self.cfg, samples)
        outputs = self.GRES_model([feature_dic])
        return feature_dic['image'], outputs

