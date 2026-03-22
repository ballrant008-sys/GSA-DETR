<div align="center">
 
<!-- <img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/1ttttttt.jpg" alt="ddq_arch" width="200"> -->
 
</div>

# A Geometric Structure-Aware Morphology–Pose Decoupled Framework for Ship Detection in Remote-Sensing Imagery

<div align="center">

<b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://openmmlab.com/">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
    <b><font size="5">GeForce RTX 4090</font></b>
    <sup>
      <a href="https://www.nvidia.cn/data-center/tesla-p100/">
        <i><font size="4">GET IT</font></i>
      </a>
    </sup>

  ![](https://img.shields.io/badge/python-3.8.16-red)
  [![](https://img.shields.io/badge/pytorch-2.1.2-red)](https://pytorch.org/)
  [![](https://img.shields.io/badge/torchvision-0.11.0-red)](https://pypi.org/project/torchvision/)
  [![](https://img.shields.io/badge/MMDetection-3.3.0-red)](https://github.com/open-mmlab/)
  
  [🛠️Installation Dependencies](https://blog.csdn.net/m0_46556474/article/details/130778016) |
  [🎤Introduction](https://github.com/open-mmlab/mmdetection) |
  [👀Download Dataset](https://pan.baidu.com/s/1ZZYeLK0vwzrXUt_AHgsn0w) |
  [🌊Aircraft Detection](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |
  [🚀Remote Sensing](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |
  [🤔End-to-End](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |

</div>

## Dependencies:

- Python 3.8.16
- [PyTorch](https://pytorch.org/) 2.1.2
- [Torchvision](https://pypi.org/project/torchvision/) 0.11.2
- [OpenCV](https://opencv.org/) 4.8.1
- Ubuntu 20.04.5 LTS
- [MMCV](https://github.com/open-mmlab/mmcv)
- 4 × GeForce RTX 4090

## Introduction

Accurate ship detection in remote sensing imagery is a key technological foundation for maritime surveillance, traffic management, and national defense. However, coupled geometric variations, loss of fine structural details, and isotropic regression bias under horizontal bounding box (HBB) supervision still hinder detection accuracy, especially under large-scale variations and complex backgrounds. To address these challenges, this paper proposes a geometric structure-aware detection transformer (GSA-DETR), which improves detection accuracy while maintaining favorable model complexity by explicitly disentangling geometric factors and enforcing structural constraints during learning. Specifically, a morphology--pose decoupled fusion (MPDF) module is proposed to construct decoupled yet complementary representations, suppressing morphology--pose coupling interference and enhancing structural consistency across scales. Meanwhile, a selective frequency reconstruction (SFR) module is devised to model and reconstruct low- and high-frequency components separately, alleviating edge smoothing induced by conventional upsampling and improving texture-detail recovery for small-scale ships. In addition, an anisotropic geometric regularization (AGR) strategy is designed to mitigate isotropic bias in HBB regression by introducing shape-dependent anisotropic constraints, thereby enhancing geometry-consistent localization robustness for elongated ships without predicting or recovering instance-level orientation. Extensive experiments on the DIOR and DOTA public datasets demonstrate that GSA-DETR consistently outperforms representative CNN- and Transformer-based detectors in detection accuracy and robustness.

<div align="center">

</div>

## Train

Single GPU training

```python
# tools/train.py
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/modelY/GSA-DETR.yaml')
    model.train(
        data='dataset/dior.yaml',
        imgsz=640,
        epochs=400,
        batch=4,
        workers=4,
        device='0',
        project='DIOR',
        name='GSA-DETR',
    )