<h1 align="center">EnvisionBOXbaby - SAM2 Installation Guide</h1>

This guide walks you through installing and running the **SAM2 (Segment Anything Model 2)** video analysis tool as part of the **EnvisionBOXbaby** toolkit. 
We will leverage this model to allow automatic object tracking and "looking at" event detection from videos

We designed this setup for **Windows ≥ 10** users.  
*(Linux is also supported by SAM2 itself, but these instructions focus on Windows.)*


### What we will need to install SAM 2: 
- **Windows ≥ 10**
- **Anaconda**:  If you do not have anaconda already installed, please donwload it here: https://www.anaconda.com/ and follow the installation instructions. 
- **Git**  
  If not installed, run in Anaconda Prompt:
  ```bash
  conda install -c anaconda git -y
  ```

### What we will install: 
- **Python ≥ 3.10**: Required to run SAM2 and its deep learning components.
- **PyTorch ≥ 2.5.1** and **torchvision**: Core deep learning libraries used by SAM2. **!!Their Version must match for CUDA support!!**
- **CUDA 12.1**: Required for GPU-accelerated segmentation.
- **SAM2**: A model from Meta AI that segments and tracks objects across video frames, and is extended in EnvisionBOXbaby to detect "looking at" events automatically.

---

##  Step 1: Create a Conda Environment for SAM2

Let's make a dedicated environment for sam2, which prevents package conflicts and ensures you use Python ≥ 3.10. 
Open your Anaconda Prompt (search for Anaconda Prompt on your machine) and type:
```bash
conda create -n sam2_env python=3.10 -y
conda activate sam2_env
```
(or substitute the sam2_env name that you prefer)
Once activated, you should see that you are working within the (sam2_env)C:\\...

>**It is important that you are working with Python ≥ 3.10**


---

##  Step 2: Install PyTorch with CUDA Support

Install PyTorch + torchvision with **CUDA 12.1** support:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

>  **It's important that the torch and torchvision versions are compatible — especially their CUDA versions.**
> Installing them together from the same PyTorch CUDA index ensures they match correctly.

> **Note:** SAM2 requires a GPU for practical performance.  
> If you don’t have an NVIDIA GPU, SAM2 may still run on CPU but will be very slow and is not officially supported.
---

## Step 3: Clone the SAM2 GitHub Repository

Navigate to the directory you want to install sam2 
```bash
cd ~
```
Clone the entire sam2 repository from facebookresearch GitHub
```bash
git clone https://github.com/facebookresearch/sam2.git
```
Navigate to the sam2 directory 
```bash
cd sam2
```

---


##  Step 4: Install SAM2 Python Dependencies

Use this command to install SAM2 in editable mode, and skip CUDA post-processing extensions:

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

What is SAM2_BUILD_CUDA=0?
It disables optional CUDA-based mask post-processing (hole filling).
Most users don’t need this feature, and disabling it avoids potential build errors on Windows

To verify the installation:

```bash
python -c "import sam2; print('SAM2 installed successfully')"
```
---



