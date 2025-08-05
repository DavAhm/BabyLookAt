<h1 align="center">EnvisionBOXbaby - Supporting Tools Installation Guide</h1>

Before running the `{script name}`, we need to install a few supporting tools and Python packages.

### What we will need:
- **sam2_env in Anaconda** (created in the [SAM2 Installation Guide](installation_SAM2.md))
- **SAM2 repository** (cloned in the [SAM2 Installation Guide](installation_SAM2.md))


## What We Will Install
- **ffmpeg** → For video frame extraction.
- **Python dependencies** (from `requirements.txt`) → For running the enhanced SAM2 UI script.
- **Model checkpoints** → Pre-trained neural network weights required by SAM2.

## 1. Install `ffmpeg`

**Why?**  
The `{script name}` uses `ffmpeg` to extract frames from videos quickly and reliably. Without it, the script will fail with “failed to extract frames” errors.


1. Download the Windows build of ffmpeg:
   - Go to: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
        (this is the most reliable source for ffmpeg installation)
   - Under **Release Builds**, download:  
     **`ffmpeg-release-full.zip`**

2. Extract the zip file:
   - For simplicity, extract to:  
     `C:\ffmpeg`

3. Add `ffmpeg` to your PATH:
   - Press **Start** → search **Environment Variables** → open **“Edit the system environment variables”**.
   - Click **Environment Variables…**
   - Under **User variables** (not System), select `Path` → **Edit** → **New**.
   - Add:
     ```
     C:\ffmpeg\bin
     ```
   - Click **OK** on all windows.

4. Verify the installation:
   - Open **Anaconda Prompt** (with your environment active).
   - Run:
     ```bash
     ffmpeg -version
     ```
   - You should see version info (no “not recognized” error).

    ###  What to do if you get an error
    If you see: 'ffmpeg' is not recognized as an internal or external command
    1. Make sure you extracted ffmpeg to the correct folder (e.g., `C:\ffmpeg`).
    2. Check that you added the **`C:\ffmpeg\bin`** path to your **User Path** environment variable:
        - Press **Start** → search **Environment Variables** → open **“Edit the system environment variables”**.
        - Click **Environment Variables…**
        - Under **User variables**, find **Path** → **Edit** → make sure `C:\ffmpeg\bin` is listed.
    3. If it is not listed, click **New** and add: `C:\ffmpeg\bin`
    4. Click **OK** on all windows and **restart Anaconda Prompt**.
    5. Try again:
        ```bash
        ffmpeg -version
        ```

---

## 2. Install All Python Packages (from `requirements.txt`)

1. Open Anaconda Prompt

2. Activate the SAM2 Environment

   ```bash
   conda activate sam2_env
   ```

   You should now see `(sam2_env)` at the beginning of your command line.

3. Navigate to the Cloned Repository
   Move to the folder where you cloned this repository:

   ```bash
   cd path\to\{repository_name}
   ```

4. Install Packages from `requirements.txt`

   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages listed in the `requirements.txt` file.

5. Verify Installation
   After installation, verify that all packages were installed correctly:

   ```bash
   pip list
   ```

   You should see the packages listed, including their versions.

---


## 3. Download the SAM2 Model Checkpoints

The SAM2 script requires **pre-trained model checkpoints** to run segmentation and object tracking. These checkpoints are large neural network weights and need to be downloaded separately.

These can be downloaded in two ways:

### Option 1: Using Git Bash

This requires **Git Bash**.

#### Check if Git Bash is available:

* Press **Start** → search for **Git Bash**.
* Or run this in the Anaconda Prompt (within the `sam2_env`):

  ```bash
  bash --version
  ```

  If you see a version number, Git Bash is installed.

If you don’t have Git Bash, you can install it here: [https://git-scm.com/downloads](https://git-scm.com/downloads)

#### Run the `download_ckpts.sh` script:

1. Navigate to the checkpoints folder:

   ```bash
   cd path/to/{repository_name}/checkpoints
   ```
2. Run the script:

   ```bash
   bash download_ckpts.sh
   ```

   This will automatically download all model checkpoints into the `checkpoints` folder.



### Option 2: Manual Download from Links

1. Download the checkpoints directly by clicking the links:

   * [sam2.1\_hiera\_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
   * [sam2.1\_hiera\_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
   * [sam2.1\_hiera\_base\_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
   * [sam2.1\_hiera\_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

2. Place them in the `checkpoints` folder of your repository:

   ```
   Object_Tracking_SAM2/
   └── checkpoints/
       ├── sam2.1_hiera_tiny.pt
       ├── sam2.1_hiera_small.pt
       ├── sam2.1_hiera_base_plus.pt
       └── sam2.1_hiera_large.pt
   ```


