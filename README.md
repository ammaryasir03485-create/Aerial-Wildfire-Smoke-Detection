# Aerial Wildfire & Smoke Early Detection System 🚁🔥

A custom-trained Computer Vision project built using **Ultralytics YOLOv8** to detect early-stage wildfires and smoke from aerial/drone imagery, preventing large-scale environmental disasters.

## 🚀 Project Overview
This system processes overhead drone or aerial views to accurately identify:
* **Fire (Flames)** 
* **Smoke (Early indicators)**

Achieving strong validation metrics (**mAP50 ~0.85+**) on a custom dataset, this project serves as a real-world automated early-warning solution.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** Ultralytics YOLOv8
* **Libraries:** OpenCV, NumPy, PyTorch

## 📂 Repository Structure
* `train.py` - Custom YOLOv8 model training script.
* `test.py` - Inference script for detecting fire/smoke on new images/videos.
* `data.yaml` - Dataset configuration file.

## 📈 Results & Performance
The model was trained for 50 epochs using standard image sizing (`imgsz=640`), yielding high accuracy across both target classes:
* **Fire mAP50:** ~85.8%
* **Smoke mAP50:** ~85.6%
