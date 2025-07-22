# TrajectoryNet++: Enhancing Shuttlecock Tracking with MSPEM and CKAM

**TrajectoryNet++** is a novel shuttlecock trajectory tracking framework designed to enhance localization accuracy under challenging video conditions. It is built on an improved **TrackNetV2** backbone, and incorporates two key modules:

* **Multi-scale Surround Prior Extraction Module (MSPEM)**
* **Cross-Knowledge Attention Module (CKAM)**

The architecture follows a two-stage pipeline:

1. **Trajectory Prediction** using an encoder-decoder model with CKAM and MSPEM enhancements.
2. **Trajectory Rectification** using an inpainting-based refinement network (**InpaintNet**) to restore occluded or missing tracking points.

---

## 🔧 Key Modules

* **📊 MSPEM**: Extracts multi-scale directional priors from early encoder stages, improving robustness to motion blur, occlusion, and background noise.

* **📈 CKAM ×3**: Three Cross-Knowledge Attention Modules are inserted into the decoder to perform progressive spatial-channel attention refinement and temporal context fusion. Each CKAM fuses attention maps from encoder stages to selectively enhance discriminative features.

* **TrackNetV2 Backbone**: The encoder-decoder structure extracts spatial-temporal features and reconstructs trajectory heatmaps, enhanced via skip connections and attention modules.

* **InpaintNet**: Learns to reconstruct missing trajectory segments based on mask supervision, enabling accurate tracking even under long-term occlusion.

---

## 🧠 Architecture Overview

<div align="center">
    <img src="./figure/NetArch.png" width="80%"/>
</div>

* **📇 Input**: Consecutive RGB video frames.
* **📊 MSPEM**: Surround prior feature extraction.
* **⬛ TrackNetV2 Encoder-Decoder**: Multi-level feature representation.
* **👥 CKAM**: Three stages of attention refinement.
* **🎯 Output**: Dense shuttlecock trajectory heatmap.

---

## 📈 Performance

TrajectoryNet++ achieves superior performance on the [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw), especially under occlusion, motion blur, and lighting variation. Comparison results:

<div align="center">
    <img src="./figure/Comparison.png" width="80%"/>
</div>

---

## 🚀 Installation

```bash
# Environment requirements
Ubuntu 16.04+
Python 3.8+
PyTorch 1.10+

# Clone the repository
git clone https://github.com/qaz812345/TrackNetV3.git
cd TrackNetV3

# Install dependencies
pip install -r requirements.txt
```

---

## 🔍 Inference

```bash
# Unzip pretrained models
unzip TrackNetV3_ckpts.zip -d ckpts

# Predict CSV trajectory from video
python predict.py \
    --video_file test.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir prediction

# Generate trajectory overlay video
python predict.py \
    --video_file test.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir prediction \
    --output_video

# For long video clips
python predict.py \
    --video_file test.mp4 \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir prediction \
    --large_video \
    --video_range 324,330
```

---

## 🏋️ Training

### 1. Dataset Preparation

Reorganize according to the structure:

```
data/
  ├── train/match1/.../*.csv, *.png, *.mp4
  ├── val/match24/...
  └── test/match1/...
```

Then run preprocessing:

```bash
python preprocess.py
```

---

### 2. Train Tracking Module (TrajectoryNet++)

```bash
python train.py \
    --model_name TrajectoryNet++ \
    --seq_len 8 \
    --epochs 30 \
    --batch_size 10 \
    --bg_mode concat \
    --alpha 0.5 \
    --save_dir exp \
    --verbose
```

---

### 3. Generate Occlusion Masks for Inpainting

```bash
python generate_mask_data.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --batch_size 16
```

---

### 4. Train InpaintNet for Trajectory Rectification

```bash
python train.py \
    --model_name InpaintNet \
    --seq_len 16 \
    --epochs 300 \
    --batch_size 32 \
    --lr_scheduler StepLR \
    --mask_ratio 0.3 \
    --save_dir exp \
    --verbose
```

---

## 📊 Evaluation

### Evaluate Full Pipeline

```bash
python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --split_list test
python test.py --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval
```

### Evaluate Tracking Only

```bash
python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval
```

### Visualize Trajectory Overlay

```bash
python test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --video_file data/test/match1/video/1_05_02.mp4
```

---

## 📉 Error Analysis

```bash
python test.py \
    --tracknet_file ckpts/TrackNet_best.pt \
    --inpaintnet_file ckpts/InpaintNet_best.pt \
    --save_dir eval \
    --output_pred

# Launch Dash-based Analysis UI
python error_analysis.py --split test --host 127.0.0.1
```

<div align="center">
    <img src="./figure/ErrorAnalysisUI.png" width="70%"/>
</div>

---

## 📚 Reference

* [TrackNetV2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)
* [Shuttlecock Trajectory Dataset](https://hackmd.io/@TUIK/rJkRW54cU)
* [Labeling Tool](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2)
