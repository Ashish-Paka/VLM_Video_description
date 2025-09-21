
# Vision-Language Cross-Embodiment Skill Alignment

> **Author:** [Ashish Paka](https://ashish-paka.netlify.app)  
> **Affiliation:** LOGOS Robotics Lab, Arizona State University  
> **Status:** Jan 2025 – Ongoing  
> **Paper Inspiration:** [HuBE: Cross-Embodiment Human-like Behavior Execution for Humanoid Robots](https://arxiv.org/html/2508.19002v1)

---

## 👤 About the Author
Robotics Research Engineer (M.S. Robotics & Autonomous Systems, ASU, GPA 3.96).  
Experience in:
- Vision-Language Models & continual learning  
- Multi-robot mapping and navigation (ROS 2)  
- Mechanical design, FEA/CFD, product lifecycle (L&T Tech Services)

> Poster @ ASU SEMTE 2025 — *Cross-Embodiment Skill Representation in Robotics*:contentReference[oaicite:0]{index=0}

---

## 🎯 Project Overview
A research platform for **learning, representing, and transferring skills across embodiments**:

- Segments demonstrations into *skill primitives* with semantic labels  
- Aligns human, video, and robot trajectories using a **Vision-Language Model**  
- Handles **morphological gaps** between source & target (bone-scaling / retargeting)  
- Supports **continual learning** & few-shot adaptation  

Inspired by HuBE’s “behavioral similarity + appropriateness + morphology” loop, but extended to skill-level reasoning and language grounding.

---

## 🏗️ System Architecture

```text
video / demos + language ─► VLM encoder
                              │
             ┌─► skill segmentation (transformers / ResNet)
             │
             ├─► cross-embodiment alignment (LoRA adapters)
             │
robot state ◄┘  ─► retargeted trajectory / policy
````

Core modules:

* `alignment/` – cross-embodiment mapping
* `segmentation/` – primitive extraction
* `vlm/` – visual + language encoders
* `adaptation/` – continual learning, LoRA

---

## 📂 Repository Layout

```bash
vlm_cross_embodiment/
├── data/          # RH20T + human/robot demos
├── src/
│   ├── vlm/
│   ├── segmentation/
│   ├── alignment/
│   └── adaptation/
├── scripts/
│   ├── preprocess_data.py
│   ├── train_alignment.py
│   ├── adapt_model.py
│   └── evaluate.py
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Ashish-Paka/vlm_cross_embodiment.git
cd vlm_cross_embodiment
pip install -r requirements.txt

# preprocess data
python scripts/preprocess_data.py --input demos/ --out data/processed

# train & evaluate
python scripts/train_alignment.py --config configs/base.yaml
python scripts/evaluate.py --robot franka --task "pick up cup"
```

> GPU (≥16 GB), ROS 2 Humble, Python 3.9+ recommended.

---

## 📊 Current Results

| Experiment         | Transfer             | Metric         | Score           |
| ------------------ | -------------------- | -------------- | --------------- |
| RH20T → FR3 Franka | Human→Robot          | Skill matching | **91.4 %**      |
| Robot A → Robot B  | Diff. arm kinematics | Traj. error    | 0.032 rad avg   |
| Continual learning | Few-shot new task    | Δ accuracy     | +74 % over base |

---

## 🔬 Research Links

* [Continual Skill & Task Learning via Dialogue (CoRL 24)](https://arxiv.org/abs/2409.03166)
* [HuBE paper](https://arxiv.org/html/2508.19002v1) – contextual & morphology-aware control
* [LOGOS Robotics Lab](https://logos-robotics-lab.github.io)


