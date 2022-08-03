## Open-Vocabulary COCO

1. Download the [COCO](https://cocodataset.org/#home) dataset.

2. Create the annotation jsons for the open-vocabulary setting. We use the scripts provide by [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb). Then add the object proposals that may cover the novel classes. You can download our pre-generated json file in [Google Drive](https://drive.google.com/file/d/1O_RU6k_s3UI74RFcpxAyIQhHmnmbRdIe/view?usp=sharing).

3. Extract the CLIP image features. You can download our pre-generated file in [Google Drive](https://drive.google.com/file/d/1nZJcr0Rl1Osy6qxbNPd1eIgZiZ0Warc6/view?usp=sharing), or use the [script](./ovdetr/scripts/save_clip_features.py) to extract it by yourself.

## Open-Vocabulary LVIS

Under preparation.
