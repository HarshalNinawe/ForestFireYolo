# Forest Fire Detection using YOLOv11

🔥 **A Production-Ready Streamlit Application for Real-Time Forest Fire and Smoke Detection**

This repository contains a complete, deployment-ready web application built using **Streamlit** and **Ultralytics YOLOv11** (`best.pt` weights). The app allows users to upload images or videos, configure inference sensitivity using intuitive sliders, and instantly detect forest fires or smoke patterns.

---

## 🚀 Features

- **Modern Responsive UI**: Custom layout featuring a sleek, dark-accented design, responsive metrics grids, and organized tab-based navigation.
- **Dynamic Settings Sidebar**:
  - **Confidence Threshold Slider** (`0.10` - `0.90`): Filter detections below a custom accuracy confidence.
  - **Intersection over Union (IoU) Slider** (`0.10` - `0.90`): Control bounding box overlaps (NMS).
  - **Model Statistics Panel**: Real-time identification of runtime hardware (CPU or GPU/CUDA) and model task.
  - **Dynamic Class Lister**: Lists classes dynamically extracted directly from the weights file.
- **High-Performance Image Detection**:
  - Support for `JPG`, `JPEG`, and `PNG` image uploads.
  - Side-by-side comparison of original and annotated images.
  - Detection speed metrics (inference time), total detections, highest, and average confidences.
  - Detailed, interactive detection tabular data displaying classes, confidence percentages, and coordinates.
  - Immediate high-resolution downloads for processed images.
- **Robust Video Detection**:
  - Support for `MP4`, `AVI`, and `MOV` video formats.
  - Frame-by-frame streaming predictions with real-time visual progress indicator (%).
  - Codec auto-fallback (`avc1`/H.264 & `mp4v`) to maximize inline browser video playback compatibility.
  - High-performance, memory-optimized processing.
  - Full-duration metric aggregations and output video download options.
- **Enterprise-Grade Stability**:
  - Cleaned temporary files automatically on startup and periodically post-processing.
  - Full check-and-catch blocks preventing app crashes on corrupt uploads or empty inputs.

---

## 📁 Folder Structure

```
ForestFireDetection/
│
├── app.py             # Streamlit entry point & user interface
├── utils.py           # Core OpenCV video pipeline, image analysis, & caching
├── best.pt            # Pre-trained YOLOv11 model weights (must be present)
├── requirements.txt   # Pip package dependencies
├── runtime.txt        # python-3.11 environment configuration
├── packages.txt       # System-level dependencies for Streamlit Cloud (libgl1)
├── .gitignore         # Config to ignore python artifacts & temporary outputs
├── README.md          # Comprehensive project manual
└── outputs/           # Destination directory for processed video clips
```

---

## 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed (tested on Python 3.10.9).

1. **Clone or Download the Repository**
   Make sure you are inside the `ForestFireDetection` project root directory.

2. **Add the Model Weights**
   Place your trained YOLOv11 model file named `best.pt` into the root directory.

3. **Create a Virtual Environment (Recommended)**
   ```bash
   # Windows PowerShell/Command Prompt
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running Locally

Start the Streamlit development server:

```bash
streamlit run app.py
```

Once running, the application will automatically open in a new tab in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

The application is completely configured for immediate deployment to **Streamlit Community Cloud** or **Hugging Face Spaces**.

### Steps for Streamlit Community Cloud:
1. Push this folder to a public GitHub repository. (Ensure `best.pt` is committed. If the model file is larger than 100MB, use Git LFS, although standard YOLOv11nano/small weights are ~5-20MB and can be pushed directly).
2. Log in to [Streamlit Share](https://share.streamlit.io/).
3. Click **New App**, then select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**. Streamlit will automatically install dependencies from `requirements.txt` and launch the app.

---

## 📦 Requirements

The app relies on the following packages:
- `streamlit`: Dashboard and UI framework.
- `ultralytics`: YOLOv11 core prediction API.
- `opencv-python-headless`: Fast frame loading and image transformations.
- `numpy`: Multi-dimensional array operations.
- `Pillow`: Image formatting and metadata loading.
- `torch` & `torchvision`: Model execution engine.

All specific versions are locked in the `requirements.txt` file.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
