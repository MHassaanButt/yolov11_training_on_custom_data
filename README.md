# 🎯 Training YOLOv11 on Custom Red Palm Weevil Dataset

![Red Palm Weevil Detection](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-brightgreen)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-orange)](https://universe.roboflow.com/enis-w5nbu/red-palm-weevil-yfykp/dataset/3)

This guide walks you through the process of training YOLOv11 on a custom dataset for detecting Red Palm Weevils. Perfect for beginners in computer vision and object detection! 🚀

## 📋 Table of Contents
- [🎯 Training YOLOv11 on Custom Red Palm Weevil Dataset](#-training-yolov11-on-custom-red-palm-weevil-dataset)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔧 Prerequisites](#-prerequisites)
  - [📦 Dataset Preparation](#-dataset-preparation)
  - [⚙️ Installation](#️-installation)
  - [🏋️‍♂️ Training](#️️-training)
  - [📊 Results Visualization](#-results-visualization)
  - [🔍 Troubleshooting](#-troubleshooting)
    - [Common Issues and Solutions](#common-issues-and-solutions)
  - [📈 Performance Tips](#-performance-tips)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgments](#-acknowledgments)

## 🔧 Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Basic understanding of command line operations

## 📦 Dataset Preparation

1. **Download the Dataset**
   ```bash
   # Clone this repository
   git clone https://github.com/MHassaanButt/yolov11_training_on_custom_data.git
   cd yolov11_training_on_custom_data
   
   # Download and extract the dataset
   wget https://universe.roboflow.com/enis-w5nbu/red-palm-weevil-yfykp/dataset/3
   unzip red-palm-weevil-yfykp.zip
   ```

2. **Verify Directory Structure**
   ```
   yourrepository/
   ├── datasets/
   │   ├── train/
   │   │   ├── images/
   │   │   └── labels/
   │   ├── valid/
   │   │   ├── images/
   │   │   └── labels/
   │   └── test/
   │       ├── images/
   │       └── labels/
   └── data_custom.yaml
   ```

3. **Create `data_custom.yaml`**
   ```yaml
   path: /path/to/yolov11_on_red_palm_weevil/datasets  # It should be absoulate path accordingly to your system directory
   train: train/images
   val: valid/images
   
   # Classes
   names:
     0: red_palm_weevil
   ```

## ⚙️ Installation

```bash
# Create and activate conda environment
conda create -n yolo11_env python=3.10
conda activate yolo11_env

# Install PyTorch (adjust according to your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Ultralytics
pip install ultralytics
```

## 🏋️‍♂️ Training

1. **Start Training**
   ```python
   from ultralytics import YOLO
   
   # Load the model
   model = YOLO("yolo11n.pt")  # nano model
   
   # Train the model
   results = model.train(
       data="/path/to/data_custom.yaml",
       epochs=100,
       imgsz=512,
       batch=16,
       device=0  # Use GPU. Use 'cpu' if no GPU available
   )
   ```

2. **Monitor Training**
   - Training progress will be displayed in the console
   - Logs and checkpoints are saved in `runs/detect/train/`

## 📊 Results Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read training results
df = pd.read_csv("runs/detect/train/results.csv")

# Create plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
sns.lineplot(data=df, x='epoch', y='metrics/mAP50(B)')
plt.title('Training Progress - mAP50')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.show()
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **FileNotFoundError for dataset**
   - Ensure all paths in `data_custom.yaml` are correct
   - Verify the dataset directory structure

2. **CUDA out of memory**
   - Reduce batch size
   - Try a smaller image size
   - Use a smaller model (e.g., YOLOv11n instead of YOLOv11x)

3. **Training not converging**
   - Increase number of epochs
   - Adjust learning rate
   - Check data quality and annotations

## 📈 Performance Tips

- Start with a small number of epochs (10-20) to ensure everything works
- Gradually increase epochs for better results
- Use image augmentation for better generalization
- Monitor validation loss to prevent overfitting

## 🤝 Contributing

Feel free to open issues or submit pull requests. All contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com/) for the dataset platform

---

Happy Training! 🎉 Remember, the key to good model performance is quality data and patient tuning. Don't hesitate to experiment and iterate!