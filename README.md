# Source-Free Open Compound Domain Adaptation in Semantic Segmentation (IEEE TCSVT)


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" height="20" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

**by [Yuyang Zhao](http://yuyangzhao.com), [Zhun Zhong](http://zhunzhong.site), [Zhiming Luo](https://sites.google.com/view/zhimingluo), [Gim Hee Lee](https://www.comp.nus.edu.sg/~leegh/), [Nicu Sebe](https://disi.unitn.it/~sebe/)**

**[[Paper]](https://ieeexplore.ieee.org/document/9785619)**

### Requirements
* Python >= 3.7
* Pytorch >= 1.7.0
* CUDA>=10.2
* Training Data
  
  Download [C-Driving Dataset](https://drive.google.com/drive/folders/1VXwbSKJnGO8etXy7H8GUjAjSIN5ddlcv?usp=sharing). Unzip the dataset and ensure the file structure is as follow:

  ```
  C-Driving
  ├── list
  ├── train
  │   ├── compound
  │   │   ├── cloudy
  │   │   ├── rainy
  │   │   └── snowy
  │   ├── open_not_used
  │   │   └── overcast
  │   └── source
  └── val
      └── compound
          ├── cloudy
          ├── rainy
          ├── snowy
          └── overcast
  ```
  **Note**: We move the overcast validation directory into the compound directory for calculating the averaged mIoU of compound and open domains.

### Run
* **Stage I**. Download [ImageNet pre-trained model](https://drive.google.com/drive/folders/1VXwbSKJnGO8etXy7H8GUjAjSIN5ddlcv?usp=sharing) and save it in `pretrain/vgg16-00b39a1b-updated.pth`
  
  * Train stage I.
    ```
    python train.py --cfg configs/train_source.yml
    ```
  * Test stage I.
    ```
    python test.py --cfg configs/train_source.yml
    ```

* **Stage II**. This stage requires the well-trained model from stage I as the pre-trained model. You can get it by testing the stage I or download [the source training model](https://drive.google.com/drive/folders/1VXwbSKJnGO8etXy7H8GUjAjSIN5ddlcv?usp=sharing) and save it in `pretrain/source_trained_model.pth`
  
  * Train stage II.
    ```
    python train.py --cfg configs/train_target.yml
    ```
  * Test stage II.
    ```
    python test.py --cfg configs/train_target.yml
    ```

<!-- You can also download the above models from [Baidu Drive](https://pan.baidu.com/s/1bYC-ohq5oR7VZ5mxQZe1Uw) (password: 6vg2). -->

### Inference

You can download the trained models in the paper from [Google Drive](https://drive.google.com/drive/folders/1VXwbSKJnGO8etXy7H8GUjAjSIN5ddlcv?usp=sharing). Change the `TEST.RESTORE_FROM` in `model_testing.yml` and then run the command
```
python test.py --cfg configs/model_testing.yml
```

### Citation
```
@article{zhao2022source,
  title={Source-free open compound domain adaptation in semantic segmentation},
  author={Zhao, Yuyang and Zhong, Zhun and Luo, Zhiming and Lee, Gim Hee and Sebe, Nicu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```

### Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [ADVENT](https://github.com/valeoai/ADVENT)
* [MixStyle](https://github.com/KaiyangZhou/mixstyle-release)
