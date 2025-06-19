[CVPR 2025] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.pdf)

Jiahao Lu and Jiacheng Deng 

<div align="center">
  <img src="figs/framework_00_rotated.png"/>
</div>

## :fire: News:
 *  Feb, 2025. Relation3D accepted by CVPR 2025.
 *  October, 2024. Relation3D achieves state-of-the-art performance in mAP, AP@50, and AP@25 on the hidden test set of ScanNetv2 ([hidden test](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d)). <br>

##  🛠️ TODO List:
- [✔] Release training and evalution code.


# Get Started

## Environment

Install dependencies and install segmentator from this [repo](https://github.com/Karbo123/segmentator).

```
# install attention_rpe_ops
cd lib/attention_rpe_ops && python3 setup.py install && cd ../../

# install pointgroup_ops
cd relation3d/lib && python3 setup.py develop && cd ../../

# install Relation3D
python3 setup.py develop

# install other dependencies
pip install -r requirements.txt
```


Note: Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (if you install cuda by conda, it won't provide nvcc and you should install cuda manually.)

## Datasets Preparation

### ScanNetv2
(1) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

(2) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
Relation3D
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

(3) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd dataset/scannetv2
python prepare_data_inst_with_normal.py.py --data_split train
python prepare_data_inst_with_normal.py.py --data_split val
python prepare_data_inst_with_normal.py.py --data_split test
```
### ScanNet200

Following [Mask3D](https://github.com/JonasSchult/Mask3D) to preprocess ScanNet200 (we only use the generated semantic labels and instance labels).

The preprocessed dataset files are organized as follows.
```
Relation3D
├── dataset
│   ├── scannet200
│   │   ├── train
│   │   │   ├── {scene:04}_{sub_scene:02}.npy
│   │   ├── val
│   │   │   ├── {scene:04}_{sub_scene:02}.npy
│   │   ├── test
│   │   │   ├── {scene:04}_{sub_scene:02}.npy
```

## Training

### ScanNetv2

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model and put into checkpoints/.
```
python3 tools/train.py configs/scannet/relation3d_scannet.yaml
```

### ScanNet200
Use the weight pretrained on scannet as the initialization (change the train.pretrain in configs/scannet/relation3d_scannet200.yaml)
```
python3 tools/train200.py configs/scannet/relation3d_scannet200.yaml
```
## Validation
### ScanNetv2
```
python3 tools/test.py configs/scannet/relation3d_scannet.yaml [MODEL_PATH] 
```
### ScanNet200
```
python3 tools/test200.py configs/scannet/relation3d_scannet200.yaml [MODEL_PATH] 
```

## Pre-trained Models


| dataset | AP | AP_50% | AP_25% |  Download  |
|---------------|:----:|:----:|:----:|:-----------:|
| [ScanNetv2](configs/scannet/relation3d_scannet.yaml) | 62.4 | 80.4 | 87.1 | [Model Weight](https://drive.google.com/file/d/1-PN43vJKaCCuzJQ7SoCmfB6d-SwnjZ34/view?usp=drive_link) |
| [ScanNet200](configs/scannet/relation3d_scannet200.yaml) | 31.6 | 41.2 | 45.6 | [Model Weight](https://drive.google.com/file/d/1nTZ2Yz3hwBtCt8u2kw_QhnkwnAVQO3lM/view?usp=sharing) |

# Citation
If you find this project useful, please consider citing:

```
@inproceedings{lu2025relation3d,
  title={Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation},
  author={Lu, Jiahao and Deng, Jiacheng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8889--8899},
  year={2025}
}
```

# Our Recent Works on 3D Point Cloud

* **SAS: Segment Any 3D Scene with Integrated 2D Priors** [\[Paper\]](https://arxiv.org/pdf/2503.08512) [\[Code\]](https://github.com/peoplelu/SAS) : The first work attempts to integrate multiple 2D scene understanding models for 3D tasks.
