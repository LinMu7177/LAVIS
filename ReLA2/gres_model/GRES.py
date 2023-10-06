from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertModel

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

from .modeling.criterion import ReferringCriterion


@META_ARCH_REGISTRY.register()
class GRES(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            test_topk_per_image: int,
            lang_backbone: nn.Module,
    ):

        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # language backbone
        self.text_encoder = lang_backbone

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        text_encoder = BertModel.from_pretrained(cfg.REFERRING.BERT_TYPE)
        text_encoder.pooler = None

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        losses = ["masks"]

        criterion = ReferringCriterion(
            weight_dict=weight_dict,
            losses=losses,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                    or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                    or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "lang_backbone": text_encoder,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        lang_emb = [x['lang_tokens'].to(self.device) for x in batched_inputs]
        lang_emb = torch.cat(lang_emb, dim=0)

        lang_mask = [x['lang_mask'].to(self.device) for x in batched_inputs]
        lang_mask = torch.cat(lang_mask, dim=0)

        lang_feat = self.text_encoder(lang_emb, attention_mask=lang_mask)[0]  # B, Nl, 768

        lang_feat = lang_feat.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        lang_mask = lang_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)

        features = self.backbone(images.tensor, lang_feat, lang_mask)
        outputs = self.sem_seg_head(features, lang_feat, lang_mask)

        res_images = torch.Tensor([]).to(self.device)
        for image in images:
            res_images = torch.cat((res_images, image.unsqueeze(0)), dim=0)

        src_masks = outputs['pred_masks']
        h, w = res_images.shape[-2:]
        src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)
        mask = torch.sigmoid(src_masks[:, 1, :, :].unsqueeze(dim=1))
        # mask = src_masks[:, 1, :, :].unsqueeze(dim=1)
        # above_threshold = src_masks[:, 1, :, :] >= 0
        # raw_mask = above_threshold.int().unsqueeze(dim=1)

        return {
            'images': res_images,
            'mask': mask
        }


        # # self.training = False
        # print('is training: ', self.training)
        # if self.training:
        #     targets = self.prepare_targets(batched_inputs, images)
        #
        #     losses = self.criterion(outputs, targets)
        #
        #     for k in list(losses.keys()):
        #         if k in self.criterion.weight_dict:
        #             losses[k] *= self.criterion.weight_dict[k]
        #         else:
        #             losses.pop(k)
        #     return losses
        # else:
        #     # concatenate outputs['images'].tensor
        #     res_images = torch.Tensor([]).to(self.device)
        #     for image in images:
        #         res_images = torch.cat((res_images, image.unsqueeze(0)), dim=0)
        #
        #     src_masks = outputs['pred_masks']
        #     h, w = res_images.shape[-2:]
        #     src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)
        #     above_threshold = src_masks[:, 1, :, :] >= 0
        #     return {
        #         'images': res_images,
        #         'mask': above_threshold.int().unsqueeze(dim=1)
        #     }

    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        for data_per_image in batched_inputs:
            # pad instances
            targets_per_image = data_per_image['instances'].to(self.device)
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            is_empty = torch.tensor(data_per_image['empty'], dtype=targets_per_image.gt_classes.dtype
                                    , device=targets_per_image.gt_classes.device)
            target_dict = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "empty": is_empty,
            }
            if data_per_image["gt_mask_merged"] is not None:
                target_dict["gt_mask_merged"] = data_per_image["gt_mask_merged"].to(self.device)

            new_targets.append(target_dict)
        return new_targets

    def refer_inference(self, mask_pred, nt_pred):
        mask_pred = mask_pred.sigmoid()
        nt_pred = nt_pred.sigmoid()
        return mask_pred, nt_pred
