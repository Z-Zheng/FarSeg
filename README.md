<h2 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h2>
<!-- <h5 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h5> -->



<h5><a href="http://zhuozheng.top/">Zhuo Zheng</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a> and Ailong Ma</h5>


<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/farseg.png"><br><br>
</div>

This is an official implementation of FarSeg in our CVPR 2020 paper [Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf).

---------------------
## News
- 2024/03, source code of FarSeg++ is released.
- 2023/10, [UV6K dataset](https://zenodo.org/record/8404754) is publcily available.
- 2023/07, FarSeg++ is accepted by IEEE TPAMI.

## Citation
If you use FarSeg or FarSeg++ in your research, please cite the following paper:
```text
@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}
@article{zheng2023farseg++,
  title={FarSeg++: Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong and Zhang, Liangpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  volume={45},
  number={11},
  pages={13715-13729},
  publisher={IEEE}
}
```

## Getting Started
### Install SimpleCV

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

#### Requirements:
- pytorch >= 1.1.0
- python >=3.6

### Prepare iSAID Dataset

```bash
ln -s </path/to/iSAID> ./isaid_segm
```

### Evaluate Model
#### 1. download pretrained weight in this [link](https://github.com/Z-Zheng/FarSeg/releases/download/v1.0/farseg50.pth)

#### 2. move weight file to log directory
```bash
mkdir -vp ./log/isaid_segm/farseg50
mv ./farseg50.pth ./log/isaid_segm/farseg50/model-60000.pth
```
#### 3. inference on iSAID val
```bash
bash ./scripts/eval_farseg50.sh
```

### Train Model
```bash
bash ./scripts/train_farseg50.sh
```


