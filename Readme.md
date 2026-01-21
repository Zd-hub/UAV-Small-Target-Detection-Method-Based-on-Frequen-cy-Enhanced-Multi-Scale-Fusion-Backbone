# UAV Small Target Detection Method Based on Frequency-Enhanced Multi-Scale Fusion Backbone 
## 🚀abstract
  Despite the widespread adoption of UAV-based object detection, traditional YOLO architectures are bottlenecked by their reliance on NMS, which complicates deployment on edge devices due to limited support across hardware acceleration platforms. While end-to-end models such as RT-DETR eliminate this bottleneck, they suffer from severe feature degradation for small targets caused by the inherent conflict between deep downsampling and detail preservation. To bridge this gap, we propose a Frequency-Enhanced Real-Time Detection framework specifically designed for UAV perspectives. Unlike standard backbones, our design incorporates a Frequen-cy-Enhanced Multi-Scale Fusion module, which transforms features into the frequency domain to explicitly amplify high-frequency components essential for small object lo-calization. Additionally, a Grouped Multi-Kernel Interaction module is introduced to dynamically capture multi-scale contextual information. Furthermore, we integrate Shape-NWD into the loss computation by introducing shape weight coefficients and scale correlation factors, directing focus toward the intrinsic attributes of bounding boxes to enhance regression accuracy for tiny targets. Experimental results on the Vis-Drone dataset demonstrate that our method improves the Average Precision by 0.9% and AP50 by 1.1% compared to the baseline, with consistent gains observed on the UAVVaste dataset
![tu](https://github.com/Zd-hub/UAV-Small-Target-Detection-Method-Based-on-Frequen-cy-Enhanced-Multi-Scale-Fusion-Backbone/blob/main/pictures/%E7%BB%93%E6%9E%84%E5%9B%BE_04.png)


## Experimental Results on the VisDrone-2019-DET Dataset

| ​**Model**​            | ​**Backbone**​         | ​**Input Size**​ | ​**Params**​ | ​**GFLOPs**​ | ​**AP**​  | ​**APs**​ |
|----------------------|---------------------|----------------|----------------|------------|---------|---------------|
| Model-R18  | ResNet18-Improved            | 640×640        | 20.8          | 79       | ​**27.5**| ​**45.9**​      |
| Model-R50 | ResNet50-Improved            | 640×640         | 42.9           | 172      | ​**29.4**| ​**47.9**​      |

---

## Experimental Results on UAVVaste Dataset

| ​**Model**​             | ​**Params**​ | ​**GFLOPs**​ | ​**AP50**​  | ​**APs**​ |
|-----------------------|----------------|------------|---------|---------------|
| Model-R18    | 20.8           | 79       | 73.1    |    36.2       |
|Model-R50    | 42.9           |172       | 74.2    | 37.3          |

## 🦄 Performance in Visdrone

![tu](https://github.com/Zd-hub/UAV-Small-Target-Detection-Method-Based-on-Frequen-cy-Enhanced-Multi-Scale-Fusion-Backbone/blob/main/pictures/%E7%BB%93%E6%9E%84%E5%9B%BE_12.png)

## 📍 Environment
- torch 1.13.1+cu11.7 
- torchvision 0.14.1+cuda11.7 
- Ubuntu 20.04
