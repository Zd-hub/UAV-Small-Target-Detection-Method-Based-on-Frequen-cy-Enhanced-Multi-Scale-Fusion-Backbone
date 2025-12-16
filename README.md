# UAV Small Target Detection Method Based on Frequency Enhanced Multi-Scale Fusion Backbone
## details
This paper proposes an RT-DETR-based object detection framework suitable for UAV perspectives, which includes a backbone network with frequency-enhanced multi-scale information fusion to address the contradiction that traditional backbones struggle to balance small target localization and multi-scale coverage.  
In addition, shape weight coefficients and scale correlation factors are introduced into the IOU loss calculation, making the loss calculation focus on the properties of the bounding box itself, amplifying the deviation weights of key directions (such as short edges) and small-scale boxes, thereby improving the performance in small target detection tasks.
<div align="center">
  <img src="https://github.com/Zd-hub/UAV-Small-Target-Detection-Method-Based-on-Frequen-cy-Enhanced-Multi-Scale-Fusion-Backbone/blob/main/picture/%E7%BB%93%E6%9E%84%E5%9B%BE.png" width=500 >
</div>

## Implementations
| Model | Input shape | Dataset | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS)
|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| Model-R18 | 640 | VisDrone2019 | 27.5 | 45.9 | 20.8 | 79 | 126|
| Model-R50 | 640 | VisDrone2019 | 29.4 | 47.9 | 42.9 | 172 | 69 |


## Performance
For complex scenarios including low illumination, dense small targets, and target occlusion, the inference results of the proposed model are presented as follows.
<div align="center">
  <img src="https://github.com/Zd-hub/UAV-Small-Target-Detection-Method-Based-on-Frequen-cy-Enhanced-Multi-Scale-Fusion-Backbone/blob/main/picture/%E7%BB%93%E6%9E%84%E5%9B%BE_12.png" width=800 >
</div>

## VisDrone-DET dataset

* trainset (1.44 GB): [BaiduYun](https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A) | [GoogleDrive](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing)
    
* valset (0.07 GB):  [BaiduYun](https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g) | [GoogleDrive](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing)
    
* testset-dev (0.28 GB): [BaiduYun](https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ) | [GoogleDrive](https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V) 
