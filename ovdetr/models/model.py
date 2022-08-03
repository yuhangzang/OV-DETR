import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        cls_out_channels=91,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, cls_out_channels)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(cls_out_channels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def get_outputs_class(self, layer, data):
        return layer(data)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ), _ = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.get_outputs_class(self.class_embed[lvl], hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
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
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class OVDETR(DeformableDETR):
    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        cls_out_channels=2,
        dataset_file="coco",
        zeroshot_w=None,
        max_len=15,
        clip_feat_path=None,
        prob=0.5,
    ):
        super().__init__(
            backbone,
            transformer,
            num_classes,
            num_queries,
            num_feature_levels,
            aux_loss,
            with_box_refine,
            two_stage,
            cls_out_channels=1,
        )
        self.zeroshot_w = zeroshot_w.t()

        self.patch2query = nn.Linear(512, 256)
        self.patch2query_img = nn.Linear(512, 256)
        for layer in [self.patch2query]:
            nn.init.xavier_uniform_(self.patch2query.weight)
            nn.init.constant_(self.patch2query.bias, 0)

        self.feature_align = nn.Linear(256, 512)
        nn.init.xavier_uniform_(self.feature_align.weight)
        nn.init.constant_(self.feature_align.bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = nn.ModuleList([self.feature_align for _ in range(num_pred)])

        self.all_ids = torch.tensor(range(self.zeroshot_w.shape[-1]))
        self.max_len = max_len
        self.max_pad_len = max_len - 3

        self.clip_feat = torch.load(clip_feat_path)
        self.prob = prob

    def forward(self, samples: NestedTensor, targets=None):
        if self.training:
            return self.forward_train(samples, targets)
        else:
            return self.forward_test(samples)

    def forward_train(self, samples: NestedTensor, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        uniq_labels = torch.cat([t["labels"] for t in targets])
        uniq_labels = torch.unique(uniq_labels).to("cpu")
        uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))][: self.max_len]
        select_id = uniq_labels.tolist()
        if len(select_id) < self.max_pad_len:
            pad_len = self.max_pad_len - len(uniq_labels)
            extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
            extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
            select_id += extra_labels.tolist()

        text_query = self.zeroshot_w[:, select_id].t()
        img_query = []
        for cat_id in select_id:
            index = torch.randperm(len(self.clip_feat[cat_id]))[0:1]
            img_query.append(self.clip_feat[cat_id][index])
        img_query = torch.cat(img_query).to(text_query.device)
        img_query = img_query / img_query.norm(dim=-1, keepdim=True)

        mask = (torch.rand(len(text_query)) < self.prob).float().unsqueeze(1).to(text_query.device)
        clip_query_ori = (text_query * mask + img_query * (1 - mask)).detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        img_query = self.patch2query_img(img_query.type(dtype))
        clip_query = text_query * mask + img_query * (1 - mask)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            _,
        ), _ = self.transformer(srcs, masks, pos, query_embeds, text_query=clip_query)

        outputs_classes = []
        outputs_coords = []
        outputs_embeds = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.get_outputs_class(self.class_embed[lvl], hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs[lvl]))

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "clip_query": clip_query_ori,
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            for temp, embed in zip(out["aux_outputs"], outputs_embed[:-1]):
                temp["select_id"] = select_id
                temp["pred_embed"] = embed
                temp["clip_query"] = clip_query_ori

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "select_id": select_id,
            }
        return out

    def forward_test(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        select_id = list(range(self.zeroshot_w.shape[-1]))
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        outputs_class_list = []
        outputs_coord_list = []
        num_patch = 15
        cache = None
        dtype = self.patch2query.weight.dtype
        for c in range(len(select_id) // num_patch + 1):
            clip_query = self.zeroshot_w[:, c * num_patch : (c + 1) * num_patch].t()
            clip_query = self.patch2query(clip_query.type(dtype))
            (
                hs,
                init_reference,
                inter_references,
                enc_outputs_class,
                enc_outputs_coord_unact,
                cache,
            ), _ = self.transformer(
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
                outputs_class = self.get_outputs_class(self.class_embed[lvl], hs[lvl])
                tmp = self.bbox_embed[lvl](hs[lvl])
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
        outputs_class = torch.cat(outputs_class_list, -2)
        outputs_coord = torch.cat(outputs_coord_list, -2)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        out["select_id"] = select_id
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
            for temp in out["aux_outputs"]:
                temp["select_id"] = select_id

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "select_id": select_id,
            }
        return out
