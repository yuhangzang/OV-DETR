<h1 align="center"> Open-Vocabulary DETR with Conditional Matching </h1>

<h2 align="center">
  <a href="https://arxiv.org/pdf/2203.11876.pdf">arXiv</a> |
  <a href="https://www.mmlab-ntu.com/project/ovdetr/index.html">Project Page</a> |
  <a href="https://github.com/yuhangzang/OV-DETR">Code</a>
</h2>

This repository contains the implementation of the following paper:
> **Open-Vocabulary DETR with Conditional Matching**<br>
> Yuhang Zang, Wei Li, Kaiyang Zhou, Chen Huang, Chen Change Loy<br>
> European Conference on Computer Vision (**ECCV**), 2022<br>
  
<p align="center">
  <img width=95% src="./assets/framework.png">
</p>

## Installation

We use the same environment as [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
You are also required to install the following packages:

- [CLIP](https://github.com/openai/CLIP)
- [cocoapi](https://github.com/cocodataset/cocoapi)
- [lvis-api](https://github.com/lvis-dataset/lvis-api)

We test our models under ```python=3.8, pytorch=1.11.0, cuda=10.1```, 8 Nvidia V100 32GB GPUs.

## Data
Please refer to [dataset_prepare.md](./dataset_prepare.md).

## Running the Model
Please refer to [run_scripts.md](./run_scripts.md).

## Model Zoo
- Open-vocabulary COCO (AP50 metric)

| Base | Novel| All | Model |
|------|------|-----|-------|
| 61.0 | 29.4 | 52.7|[Google Drive](https://drive.google.com/file/d/1_iypFgVsLQwXVrT5zDtKeFaxOcC_A3uO/view?usp=sharing)|

## Citation
If you find our work useful for your research, please consider citing the paper:
```
@InProceedings{zang2022open,
 author = {Zang, Yuhang and Li, Wei and Zhou, Kaiyang and Huang, Chen and Loy, Chen Change},
 title = {Open-Vocabulary DETR with Conditional Matching},
 journal = {European Conference on Computer Vision},
 year = {2022}
}
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgement
We would like to thanks [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [CLIP](https://github.com/openai/CLIP) and [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) for their open-source projects.

## Contact
Please contact [Yuhang Zang](mailto:zang0012@ntu.edu.sg) if you have any questions.
