# Object Tracking with SAM2

This repository provides a user-friendly interface for using [Meta AI's SAM2](https://github.com/facebookresearch/sam2) model to track objects in videos. It was developed as part of the EnvisionBOXBABY project, with a focus on analyzing infant-adult interactions using videos recorded from an infant's head-mounted camera.

[... Work in Progress...]

Creators:
- Wim Pouw (wim.pouw@donders.ru.nl): Developer
- Babajide Owoyele (email): Co-Developer
- Davide Ahmar (ahmar.davide@gmail.com): Documentation
---

##  What This Tool Does

- Lets you load a video and manually label key objects (e.g., face, toy, hand).
- Automatically tracks those objects across frames using SAM2.
- Detects **"looking at" events** by checking when a target object (e.g., infant eye region) overlaps with another object.
- Saves:
  - Annotated output video with visual feedback
  - CSV of detected "looking at" events
  - ELAN file (optional)

---

## Getting Started

1. ### Clone this GitHub Repository
  - Click the green **Code** button at the top of this page.
  - Select **Download ZIP**.
  - Extract the ZIP file to your preferred folder (e.g., `C:\Object_Tracking_SAM2`).

2. ### Install Sam2 
Follow the installation guide for SAM2: [SAM 2 Installation Instructions →](docs/installation_SAM2.md)

3. ### Install the supporting Tools and Packages 
Follow the installation guide for Tools and Packages: [Tools and Packages Installation Instructions →](docs/installation_tools_packages.md)
