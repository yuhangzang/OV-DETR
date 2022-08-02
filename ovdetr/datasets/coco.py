# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

import datasets.transforms as T
from util.misc import get_local_rank, get_local_size

from .torchvision_datasets import CocoDetection as TvCocoDetection


class CocoDetection(TvCocoDetection):
    SEEN_CLASSES = (
        "toilet",
        "bicycle",
        "apple",
        "train",
        "laptop",
        "carrot",
        "motorcycle",
        "oven",
        "chair",
        "mouse",
        "boat",
        "kite",
        "sheep",
        "horse",
        "sandwich",
        "clock",
        "tv",
        "backpack",
        "toaster",
        "bowl",
        "microwave",
        "bench",
        "book",
        "orange",
        "bird",
        "pizza",
        "fork",
        "frisbee",
        "bear",
        "vase",
        "toothbrush",
        "spoon",
        "giraffe",
        "handbag",
        "broccoli",
        "refrigerator",
        "remote",
        "surfboard",
        "car",
        "bed",
        "banana",
        "donut",
        "skis",
        "person",
        "truck",
        "bottle",
        "suitcase",
        "zebra",
    )
    UNSEEN_CLASSES = (
        "umbrella",
        "cow",
        "cup",
        "bus",
        "keyboard",
        "skateboard",
        "dog",
        "couch",
        "tie",
        "snowboard",
        "sink",
        "elephant",
        "cake",
        "scissors",
        "airplane",
        "cat",
        "knife",
    )

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
        label_map=False,
    ):
        super(CocoDetection, self).__init__(
            img_folder,
            ann_file,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size,
        )
        self._transforms = transforms
        self.cat_ids = self.coco.getCatIds(self.SEEN_CLASSES + self.UNSEEN_CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_ids_unseen = self.coco.getCatIds(self.UNSEEN_CLASSES)
        self.prepare = ConvertCocoPolysToMask(
            return_masks, self.cat2label, label_map, self.cat_ids_unseen
        )

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if len(target["labels"]) == 0:
            return self[(idx + 1) % len(self)]
        else:
            return img, target
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, cat2label=None, label_map=False, cat_ids_unseen=None):
        self.return_masks = return_masks
        self.cat2label = cat2label
        self.label_map = label_map
        self.cat_ids_unseen = cat_ids_unseen

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.label_map:
            classes = [
                self.cat2label[obj["category_id"]]
                if obj["category_id"] >= 0
                else obj["category_id"]
                for obj in anno
            ]
        else:
            classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (
            root / "train2017",
            root / "zero-shot" / f"{mode}_train2017_seen_2_proposal.json",
        ),
        "val": (root / "val2017", root / "zero-shot" / f"{mode}_val2017_all_2.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
        label_map=args.label_map,
    )
    return dataset
