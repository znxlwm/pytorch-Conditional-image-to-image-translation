# pytorch-Conditional-image-to-image-translation
Pytorch implementation of Conditional image-to-image translation [1] (CVPR 2018)
* Network architecture and parameters without information in the paper were set arbitrarily.

## Usage
```
python train.py --dataset dataset --domain_A A --domain_B B
```

### Folder structure
The following shows basic folder structure.
```
├── data
    ├── dataset # not included in this repo
        ├── A_train
            ├── aaa.png
            ├── bbb.jpg
            └── ...
        ├── A_test
            ├── ccc.png
            ├── ddd.jpg
            └── ...
        ├── B_train
            ├── eee.png
            ├── fff.jpg
            └── ...
        └── B_test
            ├── ggg.png
            ├── hhh.jpg
            └── ...
├── train.py # training code
├── utils.py
├── networks.py
└── name_results # results to be saved here
```

## Resutls
### paper results

### celebA results

## Development Environment
* NVIDIA GTX 1080 ti
* cuda 8.0
* python 3.5.3
* pytorch 0.4.0
* torchvision 0.2.1

## Reference
[1] Lin, Jianxin, et al. "Conditional image-to-image translation." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)(July 2018). 2018.

(Full paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_Conditional_Image-to-Image_Translation_CVPR_2018_paper.pdf)
