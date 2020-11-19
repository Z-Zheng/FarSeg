<h2 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h2>
<!-- <h5 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h5> -->



<h5 align="right">by <a href="http://zhuozheng.top/">Zhuo Zheng</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a> and Ailong Ma</h5>


<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/farseg.png"><br><br>
</div>

This is an official implementation of FarSeg in our CVPR 2020 paper [Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf).

---------------------


## Citation
If you use FarSeg in your research, please cite the following paper:
```text
@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
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

### Train Model
```bash
bash ./scripts/train_farseg50.sh
```


### Evaluate Model
```bash
bash ./scripts/eval_farseg50.sh
```