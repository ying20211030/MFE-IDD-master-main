# MFE-IDD-main
Multi-Scale Feature Enhanced Intelligent Defect Detection (MFE-IDD) for X-ray Weld Images A lightweight yet powerful defect detection algorithm combining CNN and Transformer architectures for high-accuracy weld inspection. 
# MFE-IDD: Multi-Scale Feature Enhanced Intelligent Defect Detection for X-ray Weld Images

This repository will host the official implementation of **MFE-IDD**, a lightweight and efficient intelligent defect detection algorithm for X-ray weld images. The model integrates CNN and Transformer architectures to achieve superior multi-scale feature extraction, with outstanding performance in both localization and classification.

## 📌 Highlights
- 🔍 93.4% mAP@50 and 91.4% mean recall
- ⚙️ Only 5.3M parameters with 57 FPS real-time detection
- 📈 Exceptional performance on small-scale pore defects
- 🏭 Well-adapted to industrial applications involving high-resolution digital radiography and digitized film

## Installation
conda create -n MFE-IDD python=3.8
conda activate MFE-IDD
pip install torch==1.13.1 torchvision==0.14.1
pip install -U openmim
mim install mmcv==2.0.0
mim install mmdet==3.1.0
pip install timm==0.4.12
pip install -r requirements.txt

## License
This project is licensed under the **Apache License 2.0**.
See the [LICENSE](LICENSE) file for more details.
