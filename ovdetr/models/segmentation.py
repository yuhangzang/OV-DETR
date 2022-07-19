# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 512], hidden_dim)

    def forward(self, samples: NestedTensor, targets=None, criterion=None):
        if self.training:
            return self.forward_train(samples, targets, criterion)
        else:
            return self.forward_test(samples)

    def forward_train(self, samples: NestedTensor, targets=None, criterion=None):
        with torch.no_grad():
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.detr.backbone(samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.detr.input_proj[l](src))
                masks.append(mask)
                assert mask is not None
            if self.detr.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.detr.num_feature_levels):
                    if l == _len_srcs:
                        src = self.detr.input_proj[l](features[-1].tensors)
                    else:
                        src = self.detr.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            max_len = 20
            uniq_labels = torch.cat([t["labels"] for t in targets])
            uniq_labels = torch.unique(uniq_labels).to("cpu")
            uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))][:max_len]
            select_id = uniq_labels.tolist()

            clip_query = self.detr.zeroshot_w[:, select_id].t()
            clip_query = self.detr.patch2query(clip_query)

            query_embeds = None
            if not self.detr.two_stage:
                query_embeds = self.detr.query_embed.weight
            (
                hs,
                init_reference,
                inter_references,
                enc_outputs_class,
                enc_outputs_coord_unact,
                _,
            ), memory = self.detr.transformer(srcs, masks, pos, query_embeds, text_query=clip_query)

            for lvl in [hs.shape[0] - 1]:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.detr.get_outputs_class(self.detr.class_embed[lvl], hs[lvl])
                tmp = self.detr.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
            out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}

            # FIXME h_boxes takes the last one computed, keep this in mind
            indices = criterion.matcher(out, targets, select_id)
            src_idx = criterion._get_src_permutation_idx(indices)
            hs_select = hs[-1][src_idx[0], src_idx[1], :]

        bbox_mask = self.bbox_attention(
            hs_select[
                None,
            ],
            memory[1],
            mask=masks[1],
        )

        seg_masks = self.mask_head(
            srcs[1], bbox_mask, [features[1].tensors, features[0].tensors, features[0].tensors]
        )
        bs = features[-1].tensors.shape[0]
        outputs_seg_masks = seg_masks.view(
            bs, len(src_idx[0]), seg_masks.shape[-2], seg_masks.shape[-1]
        )
        out["pred_masks"] = outputs_seg_masks
        out["select_id"] = select_id

        return out

    def forward_test(self, samples: NestedTensor, targets=None, criterion=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.detr.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.detr.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.detr.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        select_id = list(range(self.detr.zeroshot_w_val.shape[-1]))
        query_embeds = None
        if not self.detr.two_stage:
            query_embeds = self.detr.query_embed.weight

        outputs_class_list = []
        num_patch = 5
        bs = features[-1].tensors.shape[0]
        cache = None
        for c in range(len(select_id) // num_patch + 1):
            clip_query = self.detr.zeroshot_w_val[:, c * num_patch : (c + 1) * num_patch].t()
            clip_query = self.detr.patch2query(clip_query)
            (
                hs,
                init_reference,
                inter_references,
                enc_outputs_class,
                enc_outputs_coord_unact,
                cache,
            ), memory = self.detr.transformer(
                srcs, masks, pos, query_embeds, text_query=clip_query, cache=cache
            )

            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.detr.get_outputs_class(self.detr.class_embed[lvl], hs[lvl])
                tmp = self.detr.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_class_list.append(outputs_class)

        outputs_class = torch.cat(outputs_class_list, -2)
        prob = outputs_class[-1].sigmoid()
        scores, topk_indexes = torch.topk(prob.view(outputs_class[-1].shape[0], -1), 100, dim=1)
        labels = torch.zeros_like(prob, dtype=torch.int16).flatten(1)
        num_queries = self.detr.num_queries
        for ind, c in enumerate(select_id):
            labels[:, ind * num_queries : (ind + 1) * num_queries] = c
        labels = torch.gather(labels, 1, topk_indexes)
        select_id = torch.unique(labels).tolist()

        outputs_class_list = []
        outputs_coord_list = []
        outputs_seg_masks_list = []
        bs = features[-1].tensors.shape[0]
        cache = None
        for c in range(len(select_id) // num_patch + 1):
            select_c = select_id[c * num_patch : (c + 1) * num_patch]
            clip_query = self.detr.zeroshot_w_val[:, select_c].t()
            clip_query = self.detr.patch2query(clip_query)
            (
                hs,
                init_reference,
                inter_references,
                enc_outputs_class,
                enc_outputs_coord_unact,
                cache,
            ), memory = self.detr.transformer(
                srcs, masks, pos, query_embeds, text_query=clip_query, cache=cache
            )
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.detr.get_outputs_class(self.detr.class_embed[lvl], hs[lvl])
                tmp = self.detr.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(outputs_coords)
            outputs_class_list.append(outputs_class)
            outputs_coord_list.append(outputs_coord)

            bbox_mask = self.bbox_attention(hs[-1], memory[1], mask=masks[1])
            seg_masks = self.mask_head(
                srcs[1], bbox_mask, [features[1].tensors, features[0].tensors, features[0].tensors]
            )
            outputs_seg_masks = seg_masks.view(
                bs, self.detr.num_queries * len(select_c), seg_masks.shape[-2], seg_masks.shape[-1]
            )
            outputs_seg_masks_list.append(outputs_seg_masks)

        outputs_class = torch.cat(outputs_class_list, -2)
        outputs_coord = torch.cat(outputs_coord_list, -2)
        outputs_seg_masks = torch.cat(outputs_seg_masks_list, 1)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        out["pred_masks"] = outputs_seg_masks
        out["select_id"] = select_id

        del outputs_class_list, outputs_coord_list, outputs_seg_masks_list

        return out


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(
            k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
