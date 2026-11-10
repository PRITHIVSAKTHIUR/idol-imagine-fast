# **[Idol-Imagine-Fast](https://huggingface.co/spaces/prithivMLmods/idol-imagine-fast)**

Idol-Imagine-Fast is a dual-pipeline **Text-to-3D** and **Image-to-3D** generation studio that converts single text prompts or reference photographs into high-density 3D Gaussian splats (`.ply`). The application pairs **Idol** (`speridlabs/idol`) for 3D Gaussian splat reconstruction with **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`) for rapid 9-step text-to-image synthesis and **BRIA-RMBG-2.0** for automated foreground segmentation and bounding-box centering.

Built on Gradio with ZeroGPU support and Flash Attention 2 acceleration, the workspace provides interactive 3D splat inspection, configurable Gaussian density tiers (from `65,536` up to `1,048,576` splats), synthetic vs. real-photo conditioning modes, and over 130 ready-to-run image and text examples.

### **Key Features**

* **End-to-End Text-to-3D & Image-to-3D:** Generate a 2D concept image from text in 9 steps via `Tongyi-MAI/Z-Image-Turbo` or upload your own image, then reconstruct it into a 3D Gaussian splat via `speridlabs/idol`.
* **Automated Background Removal & Centering:** Integrates `briaai/BRIA-RMBG-2.0` to strip backgrounds, compute alpha-channel bounding boxes, crop to the subject center, and composite clean RGB tensors prior to 3D sampling.
* **Configurable Gaussian Splat Density:** Supports 5 splat count presets—`65,536`, `131,072`, `262,144` (default), `524,288`, and `1,048,576` Gaussians—exporting standard `.ply` point-cloud splat files.
* **Dual Conditioning Modes:** Switch between `Synthetic` and `Real photo` input modes alongside adjustable sampling steps, guidance scale, and time-shift (`rescale_t`) controls.
* **Auto-Zoomed Interactive 3D Viewer:** Custom client-side JavaScript hooks automatically adjust the 3D viewport orbit radius on `.ply` load while preserving session credentials across file fetches.

### **Repository Structure**

```text
├── example-images/
│   ├── A (1).webp
│   ├── ...
│   └── A (71).webp
├── idol/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── birefnet.py
│   │   ├── decoder.py
│   │   ├── dit.py
│   │   └── layers.py
│   ├── __init__.py
│   ├── camera.py
│   ├── conditioning.py
│   ├── gaussian.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── sample.py
│   └── sampler.py
├── app.py
├── LICENSE.txt
├── model.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the `idol-imagine-fast` environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU with `bfloat16` support is required.

* **Python Version:** Minimum Python **3.12.12** is required; Python **3.14** is recommended.
* **PyTorch Version:** `torch==2.13.0` or above is required for optimal compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended, matching the execution environment on the live Hugging Face Space.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on `uv.lock`.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast.git
cd idol-imagine-fast
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the application**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Clone the Repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast.git
cd idol-imagine-fast
```

**2. Create a Virtual Environment & Install Core Dependencies**
Install the primary deep learning stack, 3D Gaussian splat utilities, and Gradio components listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Launch the Application**

```bash
python app.py
```

#### **Core Requirements List (`requirements.txt`)**

```text
torch==2.13.0
torchvision==0.28.0
transformers>=4.56
diffusers>=0.36
huggingface_hub>=0.34
safetensors>=0.4
tensordict>=0.8
numpy>=1.26
pillow>=10.0
plyfile>=1.0
tqdm>=4.66
rerun-sdk==0.42.0
gradio-rerun==0.42.0
gradio==6.31.0
```

### **Usage**

Once the Gradio server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Provide or Generate an Input Image:**
* **Text-to-Image-3D Tab:** Enter a descriptive prompt (e.g., *"A futuristic Plane 3D"*) and click **1.Generate Image**. `Z-Image-Turbo` will synthesize the image and automatically run background removal and centering via `BRIA-RMBG-2.0`.
* **Image-to-3D Tab:** Upload an image directly, paste from your clipboard, or click any of the 71 thumbnails in the **Image Examples** gallery below.


2. **Select Input Type & Settings:**
* Choose **Synthetic** or **Real photo** under **Input type**.
* Optionally expand **Settings** to select the target **Gaussians** count (`65,536` to `1,048,576`), adjust **Steps** (default `50`), **Guidance scale** (default `3.0`), and **Time shift** (`3.0`).


3. **Generate 3D Splat:** Click **2.Generate 3D**. The reconstructed `.ply` Gaussian splat will load directly into the interactive 3D viewer for orbiting, zooming, and downloading.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast.git](https://github.com/PRITHIVSAKTHIUR/idol-imagine-fast.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/idol-imagine-fast](https://huggingface.co/spaces/prithivMLmods/idol-imagine-fast)
