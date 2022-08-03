We use the same scripts as [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
If you are not familiar with DETR series papers, you are recommend to first read the documents of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [DETR](https://github.com/facebookresearch/detr).

---
For example, to train the model on single node and 8 GPUs:

```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --dataset_file coco \
    --coco_path xxxx/COCO/ \
    --output_dir ./output/ \
    --num_queries 300 \
    --with_box_refine \
    --two_stage \
    --label_map \
    --max_len 15 \
    --prob 0.75 \
    --clip_feat_path xxxx/clip_feat_coco.pkl \
```

The meaning of these arguments:
* `dataset_file`: `coco` or `lvis`.
* `coco_path` / `--lvis_path`: the dataset directory path.
* `label_map`: mapping the default categories ids (e.g, 0-91 for COCO) to a contiguous array (e.g, 0-64 for the open-vocabulary COCO setting). 
* `output_dir`: path to the log and checkpoint files.
* `max_len`: the symbol `R` in the paper to control the repeat times for object queries. You are recommended to use large value to reduce the convergence time.
* `prob`: the probability of selecting CLIP text or image features for conditional matching. `prob=1.0` refers to merely use the CLIP text embeddings.
* `clip_feat_path`: path to the pre-computed file of CLIP image features.

---
To evaluate the model, you need add two arguments, `--eval` and `--resume` (same as Deformable DETR):
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --dataset_file coco \
    --coco_path xxxx/COCO/ \
    --output_dir ./output/ \
    --num_queries 300 \
    --with_box_refine \
    --two_stage \
    --label_map \
    --eval \
    --resume xxx/checkpoint.pth \
```

* `resume`: path to the checkpoint file.

---
To train on the instance segmentation task, you need add two arguments, `--masks` and `--frozen_weights` (same as DETR):
```
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --dataset_file coco \
    --coco_path xxxx/COCO/ \
    --output_dir ./output/ \
    --num_queries 300 \
    --with_box_refine \
    --two_stage \
    --label_map \
    --masks \
    --frozen_weights xxx/checkpoint.pth \
```

* `frozen_weights`: path to the pretrained model.