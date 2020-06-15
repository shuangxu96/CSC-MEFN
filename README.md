# CSC-MEFN
Deep Convolutional Sparse Coding Networks for Image Fusion. [arxiv](https://arxiv.org/abs/2005.08448)

[Shuang Xu *](https://xsxjtu.github.io/), [Zixiang Zhao *](https://www.researchgate.net/profile/Zixiang_Zhao5/), [Yicheng Wang](https://www.researchgate.net/profile/Wang_Yicheng4), Kai Sun, [Chunxia Zhang](https://www.researchgate.net/profile/Chun_Xia_Zhang/), [Junmin Liu](http://gr.xjtu.edu.cn/web/junminliu/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang/). (* equal contributions)

## Requirements
- pytorch (my version is 1.3.0)
- kornia
- tensorboardX

## Train & Test
### Retrain and Test CSC-MEFN
The train and test codes are available lines 7-93 and 96-103 of train_MMF.py. If you want to retrain this network, you should:
- Please download and unzip the [dataset](https://mega.nz/folder/LQwVhZ4J#PNGzSnjkrqjPD4M7Td2jMA) into the folder `MEF_data`. My folder is organized as follows:
```
    mypath
    ├── train
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    ├── tvalidation
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    ├── test
    │   ├── 1
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
    │   ├── 2
    │     ├── 1.JPG
    │     ├── 2.JPG
    │     └── ...
```

- Run lines 7-93 for training.
- Run lines 96-103 for testing.
