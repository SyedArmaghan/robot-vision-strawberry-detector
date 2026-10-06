# 🍓Strawberry Detection & Ripeness Classification

An edge-optimized computer vision pipeline developed in Python with OpenCV for autonomous agricultural harvesting robots. The system suppresses foliage and greenhouse background noise using vegetation-difference indexing, localizes individual fruit centroids for robotic end-effector targeting, and classifies maturity into **Ripe**, **Semi-Ripe**, and **Raw (Unripe)**.

---

## 🚀 Key Highlights
- **Foliage Suppression Index:** Uses color-difference transformation ($R - G$) to suppress green leaves and white hydroponic gutters without heavy GPU dependencies.
- **Ripeness Classification:** Multi-stage inspection combining excess red signal intensity with LAB color space analysis for pale/unripe fruits.
- **Robotics-Ready Output:** Computes bounding boxes and exact centroid coordinates `(cx, cy)` formatted for microcontrollers/actuators via UART/Serial.
- **Batch Processing:** Processes entire image directories and outputs annotated frames and yield statistics.

---

## 📦 Installation

```bash
git clone [https://github.com/SyedArmaghan/robot-vision-strawberry-detector.git](https://github.com/SyedArmaghan/robot-vision-strawberry-detector.git)
cd robot-vision-strawberry-detector
pip install -r requirements.txt
