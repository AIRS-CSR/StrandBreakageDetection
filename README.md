# A Real-Time Strand Breakage Detection Method for Power Line Inspection with UAVs

This is the official implementation for [A Real-Time Strand Breakage Detection Method for Power Line Inspection with UAVs](https://doi.org/10.3390/drones7090574)

## Abs

This repository contains the source codes of our paper "A Real-Time Strand Breakage Detection Method for Power Line Inspection with UAVs" , and the part of the power line segmentation dataset used in the experiments. The codes include three parts: 

1) The overall pipe line of the strand breakage detection method proposed in the paper, which integrages the procedures of power line segmentation network inferece, power line extraction, power line image patch cropping, and image patch classification network inference ;
2) The power line segmentation network, BA-NetV2;
3) The image patch classification network based on mult-task learning.
4) The power line segmentation dataset open-sourced here is reproduced from the open-source dataset PLDU and PLDM proposed by Zhang et al. in their paper "Detecting power lines in UAV images with convolutional features and structured constraints" published in Remote Sensing, 2019. We conducted pixel-wise annotation in the image segmentation style. Each part of the code contains a ReadMe and/or a Requirements file telling how to use the code.

## Data

lf you want to compare your results with ours, you may download them from here [Baidu Drive](https://pan.baidu.com/s/1LIwyJVspP3A33EmTxe8yWw)[ikcp] or [Google Drive](https://drive.google.com/file/d/1tO1d4ZOECfcFu8QYYGQavkdTjYej8AkW/view?usp=drive_link).

## Cite

Please cite our paper when using the codes or data in this repository:
```
@article{yan2023real,
  title={A Real-Time Strand Breakage Detection Method for Power Line Inspection with UAVs},
  author={Yan, Jichen and Zhang, Xiaoguang and Shen, Siyang and He, Xing and Xia, Xuan and Li, Nan and Wang, Song and Yang, Yuxuan and Ding, Ning},
  journal={Drones},
  volume={7},
  number={9},
  pages={574},
  year={2023},
  publisher={MDPI}
}
```


