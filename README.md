# DeepSSIM
This is the repository of paper “Structural Similarity in Deep Features: Unified Image Quality Assessment Robust to Geometrically Disparate Reference”. 
## Abstract

Image Quality Assessment (IQA) with references plays an important role in optimizing and evaluating computer vision tasks. Traditional methods assume that all pixels of the reference and test images are fully aligned. Such Aligned-Reference IQA (AR-IQA) approaches fail to address many real-world problems with various geometric deformations between the two images. Although significant effort has been made to attack Geometrically-Disparate-Reference IQA (GDR-IQA) problem, it has been addressed in a task-dependent fashion, for example, by dedicated designs for image super-resolution and retargeting, or by assuming the geometric distortions to be small that can be countered by translation-robust filters or by explicit image registrations. Here we rethink this problem and propose a unified, non-training-based Deep Structural Similarity (DeepSSIM) approach to address the above problems in a single framework, which assesses structural similarity of deep features in a simple but efficient way and uses an attention calibration strategy to alleviate attention deviation. The proposed method, without application-specific design, achieves state-of-the-art performance on AR-IQA datasets and meanwhile shows strong robustness to various GDR-IQA test cases. Interestingly, our test also shows the effectiveness of DeepSSIM as an optimization tool for training image super-resolution, enhancement and restoration, implying an even wider generalizability.

## DeepSSIM code

We provide two implementations of DeepSSIM for evaluation purposes: a MATLAB version (recommend) and a PyTorch version. We also provide DeepSSIM-Lite (PyTorch version) for optimization.

### 1. Matlab version 
#### Enviroment
MATLAB R2022b
#### Usage
Run: ./DeepSSIM_matlab/ DeepSSIM_demo.m

### 2. PyTorch version 
#### Enviroment
python 3.7,
pytorch 1.13.1
#### Usage
Run: ./DeepSSIM_pytorch/ DeepSSIM_demo.py

## Citation
If you use our code, or our work is useful for your research, please consider citing: 
```
coming soon
```  
If you have any questions, please feel free to contact kekezhang1102@163.com (recommend), zhangkeke@htu.edu.cn, kekezhang112@gmail.com.
