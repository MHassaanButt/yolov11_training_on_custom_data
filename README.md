# 🎯 Training YOLOv11 on Custom Red Palm Weevil Dataset

![Red Palm Weevil Detection](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-brightgreen)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-orange)](https://universe.roboflow.com/enis-w5nbu/red-palm-weevil-yfykp/dataset/3)

Welcome to the world's most exciting repository about detecting buff insects! 🏋️‍♂️🐞 This guide walks you through the process of training YOLOv11 to detect Red Palm Weevils - because someone has to keep an eye on these gym-enthusiast bugs!

## 🎭 Why Red Palm Weevils?

Let's be honest - when you dreamed of becoming a computer vision expert, you probably imagined detecting exotic sports cars or cute puppies. Instead, here you are, training a state-of-the-art AI model to find insects that look like they've been hitting the gym a bit too hard. But fear not! By the end of this guide, you'll be the world's foremost expert in digital weevil detection. It's not much, but it's honest work.

> "Before this model, I had to detect red palm weevils manually. Do you know how hard it is to get them to stand still for photos?" - A Very Patient Entomologist

## 📋 Table of Contents
- [Prerequisites](#prerequisites) (Spoiler: Patience and a sense of humor)
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [Bug Debugging (Get it?)](#troubleshooting)

## 🔧 Prerequisites
- Python 3.8 or higher (Weevils are picky about their Python versions)
- CUDA-capable GPU (These buff bugs need some serious computing power)
- Basic understanding of command line operations
- Ability to not laugh when telling people you're training an AI to find buff insects

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
yolov11_training_on_custom_data/
├── datasets/
│   ├── train/
│   │   ├── images/  # Weevil glamour shots
│   │   └── labels/  # Weevil name tags
│   ├── valid/
│   │   ├── images/  # Weevil validation photos
│   │   └── labels/
│   └── test/
│       ├── images/  # Weevil final exam photos
│       └── labels/
└── data_custom.yaml  # Weevil configuration file
```

## ⚙️ Installation

```bash
# Create and activate conda environment (Weevil-friendly zone)
conda create -n yolo11_env python=3.10
conda activate yolo11_env

# Install PyTorch (The heavy lifting equipment)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Ultralytics (The weevil detection toolkit)
pip install ultralytics
```

## 🏋️‍♂️ Training

1. **Start Training**
```python
from ultralytics import YOLO

# Load the model (Time to pump some pixels!)
model = YOLO("yolo11n.pt")  # 'n' for 'numerous weevils'

# Train the model
results = model.train(
    data="/path/to/data_custom.yaml",
    epochs=100,  # Give those weevils a proper workout
    imgsz=512,   # Weevils like their images like their protein shakes - large
    batch=16,    # Number of weevils to spot at once
    device=0     # Use GPU. CPU might get scared of the buff bugs
)
```

## 📊 Results Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read training results (Weevil Workout Logs)
df = pd.read_csv("runs/detect/train/results.csv")

# Create plot (Weevil Progress Chart)
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")  # Weevils prefer a clean gym
sns.lineplot(data=df, x='epoch', y='metrics/mAP50(B)')
plt.title('Training Progress - How Buff Is Our Model?')
plt.xlabel('Epoch (Sets)')
plt.ylabel('mAP50 (Muscle Achievement Progress)')
plt.show()
```

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **FileNotFoundError for dataset**
   - The weevils are camera shy. Make sure your paths are correct.
   - Check if the weevils haven't moved to a different directory for better lighting.

2. **CUDA out of memory**
   - Your GPU is intimidated by the weevils. Try:
     - Reducing batch size (Less weevils per set)
     - Using a smaller image size (Put the weevils on a diet)
     - Switching to a smaller model (Not all heroes wear capes or need large models)

3. **Training not converging**
   - Weevils are stubborn. Try:
     - More epochs (More reps)
     - Adjusting learning rate (Change the workout intensity)
     - Checking data quality (Make sure your weevils are properly labeled)

## 🎓 Graduation

Congratulations! You're now officially a Weevil Detection Expert. Your parents must be so proud.

### Project Roadmap
- [x] Teach AI to detect weevils
- [x] Wonder why we're teaching AI to detect weevils
- [ ] Expand to other gym-enthusiast insects
- [ ] Create WeevilGPT (it just makes bug puns)

## 🤝 Contributing

Found a way to make our weevil detection even buffer? Contributions are welcome! Just be gentle with the weevils, they're sensitive despite their tough exterior.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The weevils have their own union agreement.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com/) for the dataset platform
- All the weevils who posed for our dataset

---

*No weevils were harmed in the making of this model. They were all too buff to be harmed anyway.* 🏋️‍♂️🐞

Happy Weevil Hunting! 🎉 Remember, in the grand scheme of things, we're all just trying to make the world a better place, one weevil detection at a time. Don't let your dreams be dreams - even if those dreams are oddly specific about detecting buff insects.