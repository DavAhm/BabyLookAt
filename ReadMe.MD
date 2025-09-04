#  EnvisionObjectAnnotator — Object Tracking & “Looking At” Detection with SAM2

![Demo](https://raw.githubusercontent.com/DavAhm/EnvisionObjectAnnotator/main/extra/Video_Process.gif)

This repository provides a user-friendly Python application built on [Meta AI’s SAM2](https://github.com/facebookresearch/sam2) model for **object tracking and overlap (“looking at”) detection** in videos.  

The tool was developed as part of the **EnvisionBOXBABY project**, with a focus on analyzing infant–adult interactions using videos recorded from an infant’s head-mounted camera. However, it can be used for **any scenario** where you want to annotate objects and detect when one *target* object overlaps with others.

---

##  Features

- 🖼️ **Interactive annotation**: select a reference frame, click to add positive/negative points, and name each object.
- 🎯 **Target detection**: any object named with `"target"` (case-insensitive) is treated as the gaze/marker object.
- 🔍 **Event detection**: logs “looking at” events whenever the target overlaps another object:
  - By pixel overlap above a threshold  
  - Or by centroid inclusion  
- 📂 **Outputs**:
  - Annotated **video** with masks and status overlays
  - Frame-by-frame **CSV** with bounding boxes, centroids, overlap info
  - Time-aligned **ELAN (.eaf)** file for qualitative coding

---

## Getting Started

### 1. Clone this repository
Click the green **Code** button (top right) → **Download ZIP** → extract it to a folder (e.g., `C:\EnvisionObjectAnnotator`).  
Or use git:
```bash
git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git
cd EnvisionObjectAnnotator

