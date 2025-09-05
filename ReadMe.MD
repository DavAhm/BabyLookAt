<h1 align="center" style=font-size:200px>EnvisionObjectAnnotator:</h1>
<h2 align="center" style=font-size:200px>An Automatic Object-to-Object Overlap Detector with SAM2</h2>

 <p align="center" style=bold [EnvisionBox Module](https://envisionbox.org/embedded_EnvisionObjectAnnotator.html)</p>

*Authors*
Wim Pouw         (wim.pouw@donders.ru.nl)         
Babajide Owoyele (babajide.owoyele@hpi.de)        
Davide Ahmar     (ahmar.davide@gmail.com)

<p align="center">
  <img src="extra/Video_Process.gif">
</p>

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

##  Getting Started

### 1. Clone this repository
Click the green **Code** button (top right) → **Download ZIP** → extract it to a folder (e.g., `C:\EnvisionObjectAnnotator`).  
Or use git:
```bash
git clone https://github.com/DavAhm/EnvisionObjectAnnotator.git
cd EnvisionObjectAnnotator
```
2. ### Install Sam2 
Follow the installation guide for SAM2: [SAM 2 Installation Instructions →](docs/installation_SAM2.md)

3. ### Install the supporting Tools and Packages 
Follow the installation guide for Tools and Packages: [Tools and Packages Installation Instructions →](docs/installation_tools_packages.md)

---

##  How It Works

1. **Load your video** → supports `.mp4`, `.mov`, `.avi`, etc.  
2. **Pick a reference frame** → usually frame `0`.  
3. **Annotate objects**:  
   - Left-click = positive point  
   - Right-click = negative point  
   - Press **C** to name the object (must contain `"target"` for gaze markers)  
   - Press **T** to test masks  
   - Press **Enter** when done  
4. **Set detection threshold** → default is 10% overlap.  
5. **Process video** → masks are propagated, overlaps are detected, and outputs are generated.

---

##  What it outputs:

- **Annotated video**: shows objects with color-coded masks and on-screen event labels  
- **CSV file**: frame-by-frame details with bounding boxes, centroids, areas, and overlaps  
- **ELAN file**: time-aligned tiers with “Looking at: [object]” events for qualitative coding

---

## Citation
If you use this tool in your research, please cite the EnvisionBox project.  
*(Full reference will be added when publication is available.)*


---

##  Related Resources
- [Meta AI SAM2](https://github.com/facebookresearch/sam2)  
- [EnvisionBox Project](https://www.envisionbox.org)  

