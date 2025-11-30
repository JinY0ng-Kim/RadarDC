# RadarDC
Accurate dense depth completion using sparse and elevation-ambiguous radar data combined with RGB images.

### Our Key Contributions
- Leveraging a robust depth foundation model for accurate dense depth completion.
- Learning radar elevation features by exploiting similarity with MDE predictions.
- Empirically demonstrating that RadarDC achieves superior performance compared to existing state-of-the-art methods.


**Monocular Depth Estimation Model:** [UniDepth V2](https://github.com/lpiccinelli-eth/UniDepth.git)

### Qualitative Result
<img src="figure/qualitative" alt="cover" style="zoom:50%;" />

### Quantitative Results
| Distance | Method                         | RMSE (mm) ↓ | MAE (mm) ↓ |
|----------|--------------------------------|-------------|------------|
| 80 m     | RC-PDA (CVPR 2021)             | 7692.8      | 3713.6     |
| 80 m     | RadarNet (CVPR 2023)           | 4898.7      | 2179.3     |
| 80 m     | Sparse-Beats-Dense (ECCV 2024) | 4609.6      | 1927.0     |
| 80 m     | **Ours**                        | **4565.6**  | **1889.7**     |

## 🔗Pre-trained Weight
[Google Drive Link](https://drive.google.com/file/d/1qVLrhaTNYOMJ1MXi_OkZwwhwPx0jxEqn/view?usp=sharing)
```
RadarDC
├── checkpoints
│   ├── RadarDC_V6_best.pth
```

## Dataset
**NuScenes Dataset:** [NuScenes](https://www.nuscenes.org/nuscenes)

- We used the official NuScenes data split following RadarNet’s data loading approach.  
- Thanks to RadarNet: [Radar-Camera Fusion Depth](https://github.com/nesl/radar-camera-fusion-depth.git)
```
bash _Scripts/dataset_nuScenes_Train.sh
bash _Scripts/dataset_nuScenes_Test.sh
```

```
RadarDC
├── data
│   ├── nuscenes
│   ├── nuscenes_derived_ALL
│   ├── nuscenes_derived_test_ALL
├── data_dervied
│   ├── training
│   ├── testing
│   ├── validation
│   ├── dataset_nuScenes_Train.py
│   ├── dataset_nuScenes_Test.py
```

## Usage
### Environment Setup
```
conda create -n radardc python=3.10.18
conda activate radardc
pip install -r requirements.txt
```
Please follow the instructions in the UniDepth V2 repository to complete the setup:  
https://github.com/lpiccinelli-eth/UniDepth.git

### Training
We trained our model on 8 H200 GPUs  
```
torchrun --nproc_per_node=8 train.py
```
### Evaluation
```
torchrun --nproc_per_node=8 test.py
```


