# AI Image Enhancer 🚀

An AI-assisted image enhancement system built with Python and OpenCV.

The project analyzes an input image, determines what kind of enhancement it needs using a rule-based decision system, and then applies the required enhancement operations automatically.

> ⚠️ **Status:** Work in Progress  
> The core analysis → rules → enhancement pipeline is working, but enhancement quality and decision thresholds are still being tuned.

---

## 📌 Project Overview

The goal of this project is to build an intelligent image enhancement pipeline that does not blindly apply every enhancement to every image.

Instead, the system first analyzes the image and then decides what enhancement operations are actually required.

### Current Pipeline

```text
Input Image
     ↓
Image Preprocessing
     ↓
Image Analysis
     ↓
Enhancement Rules
     ↓
Enhancement Engine
     ↓
Post-Processing
     ↓
Enhanced Image