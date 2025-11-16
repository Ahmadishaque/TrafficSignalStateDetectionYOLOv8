# 🚦 Traffic Signal State Detection using Fine-Tuned YOLOv8  
*Real-time small-object detection for Intelligent Transportation Systems*

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)]()
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## 🧾 Abstract
Small-object detection remains a persistent challenge in computer vision, particularly within urban driving scenarios where traffic lights occupy minimal pixel space.  
This work fine-tunes the **YOLOv8-Large** model on a custom dataset of traffic light states to achieve robust, high-speed detection of **five distinct classes** — *Red, Yellow, Green, Wait-On, and Off*.  
The model achieves a **mean Average Precision (mAP50–95) of ~0.46** and operates at **~45 FPS**, demonstrating an effective trade-off between accuracy and real-time performance for deployment in **intelligent transportation systems (ITS)**.

---

## 🧩 Problem Statement
Urban traffic systems rely on accurate signal recognition to ensure vehicular safety and efficient flow.  
However, environmental variability (illumination, rain, occlusion) and small object size make traditional vision approaches unreliable.  
This project addresses these challenges by leveraging **transfer learning** with a **fine-tuned YOLOv8** detector optimized for **multi-class traffic signal state recognition**.

---

## ⚙️ Technical Stack

| Component | Technology |
|------------|-------------|
| **Framework** | Ultralytics YOLOv8 |
| **Language** | Python |
| **Core Libraries** | PyTorch, OpenCV, NumPy, Matplotlib, tqdm |
| **Hardware Used** | NVIDIA A100 GPU |
| **Model File** | `traffic_light_model.pt` |
| **Output Video** | `inferred_traffic_light_video.avi` |

---

## 🧠 Methodology

### 1. Dataset Preparation & Visualization
- Utilized *Small Traffic Light Dataset* (YOLO format, 5 classes).  
- Visualized annotated bounding boxes for qualitative verification.  
- Ensured balanced sampling across traffic signal states.

### 2. Model Training
- Base model: `yolov8l.pt` (pretrained on COCO).  
- Fine-tuned for **120 epochs** with a **batch size of 32** and **768×768** resolution.  
- Optimizer: **AdamW** with automatic learning rate adjustment.  
- Validation metric: **mAP50–95 ≥ 0.46**.

### 3. Inference & Evaluation
- Performed inference on unseen test video sequences.  
- Annotated detections with bounding boxes and confidence levels.  
- Achieved **real-time inference (~45 FPS)** with stable frame-by-frame classification.

---

## 📊 Performance Metrics

| Metric | Score |
|--------|--------|
| Precision | 0.96 |
| Recall | 0.74 |
| mAP@50 | 0.81 |
| mAP@50–95 | **0.46** |
| FPS | ~45 |

---

## 🧪 Results and Discussion
- The model successfully generalizes to **varying weather and lighting conditions**.  
- Performs well on **small object scales**, validating YOLOv8’s feature pyramid and transformer-based backbone.  
- The results indicate strong potential for **edge deployment** in embedded traffic control systems.

---

## 🔬 Research Contributions
1. Demonstrated **transfer learning** of YOLOv8 for a domain-specific, small-object dataset.  
2. Achieved **real-time inference** without compromising detection accuracy.  
3. Developed an interpretable pipeline for **multi-class signal state detection**.  
4. Highlighted the potential of deep vision models in **intelligent transportation systems (ITS)**.

---

## 🧰 Repository Structure

```bash
├── dataset/
│   ├── train/
│   ├── valid/
├── traffic_light_model.pt
├── inferred_traffic_light_video.avi
├── trafficlightProject.pdf
├── main.ipynb
└── README.md
```


---

## 🎥 Demonstration
- **Output Video:** `inferred_traffic_light_video.avi`  
- **Trained Model:** `traffic_light_model.pt`

---

## 🚀 Future Work
- Integrate **RT-DETR** and **SAHI** for enhanced small-object performance.  
- Extend to **multi-modal perception** (e.g., camera + radar fusion).  
- Deploy on **edge AI devices** (NVIDIA Jetson / Raspberry Pi 5).  
- Explore **quantization and pruning** for optimized embedded performance.

---

## 🧾 Citation

```bibtex
@project{TrafficSignalYOLOv8,
  title   = {Traffic Signal State Detection using Fine-Tuned YOLOv8},
  author  = {Ahmad Ishaque Karimi},
  year    = {2025},
  note    = {GitHub repository: https://github.com/Ahmadishaque/TrafficSignalStateDetectionYOLOv8}
}
```



---

## 👨‍💻 Author

**Ahmad Ishaque Karimi**  
Graduate Student — Data Science & Computer Vision Research  
📧 ahmadishaquekarimi@gmail.com  
🔗 [LinkedIn]([https://www.linkedin.com/in/your-profile])

---

## 🧠 Keywords
`Object Detection` · `YOLOv8` · `Traffic Light Detection` · `Computer Vision` · `Deep Learning` · `Intelligent Transportation Systems`

---
