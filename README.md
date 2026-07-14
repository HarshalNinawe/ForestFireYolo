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
  - High-performance, memory-optimized processing.
  - Full-duration metric aggregations and output video download options.
- **Enterprise-Grade Stability**:
  - Temporary files automatically cleaned up after each prediction.
  - Full exception handling preventing app crashes on corrupt uploads or empty inputs.

---

## 📁 Folder Structure

```
ForestFireDetection/
│
├── app.py             # Streamlit entry point & user interface
├── utils.py           # Core OpenCV video pipeline, image analysis, & caching
├── best.pt            # Pre-trained YOLOv11 model weights (must be present)
├── requirements.txt   # Pip package dependencies
├── Dockerfile         # Production Docker image configuration
├── .dockerignore      # Docker build context exclusions
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── outputs/           # Temporary directory for processed video clips
    └── .gitkeep
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Docker (for deployment)

### Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/HarshalNinawe/ForestFireYolo.git
   cd ForestFireYolo
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Locally**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`.

---

## 🐳 Docker Deployment

### Build and Run Locally with Docker

```bash
docker build -t forest-fire-detection .
docker run -p 8501:8501 forest-fire-detection
```

The app will be available at `http://localhost:8501`.

---

## ☁️ Deploy on Render.com

### Step-by-Step Instructions

1. **Push to GitHub**
   Ensure all files (including `best.pt` and `Dockerfile`) are committed and pushed to your GitHub repository.

2. **Create a Render Account**
   Sign up at [render.com](https://render.com/) and connect your GitHub account.

3. **Create a New Web Service**
   - Click **New** → **Web Service**.
   - Select your `ForestFireYolo` repository.
   - Render will auto-detect the `Dockerfile`.

4. **Configure the Service**
   | Setting | Value |
   | :--- | :--- |
   | **Name** | `forest-fire-detection` |
   | **Region** | Choose nearest |
   | **Instance Type** | Standard or higher (needs ~1GB RAM minimum) |
   | **Docker Command** | *(leave default — uses Dockerfile CMD)* |

5. **Deploy**
   Click **Create Web Service**. Render will build the Docker image and deploy your app.

6. **Access Your App**
   Once deployed, Render provides a public URL like:
   ```
   https://forest-fire-detection.onrender.com
   ```

### Important Notes
- **Model File Size**: `best.pt` (~5 MB) is within GitHub's file size limit and will be included in the Docker build.
- **Free Tier**: Render's free tier may spin down after inactivity. The first request after idle may take 30-60 seconds.
- **Memory**: For video processing, use a Standard instance ($7/month) or higher for reliable performance.

---

## 📦 Requirements

| Package | Purpose |
| :--- | :--- |
| `streamlit` | Web application framework |
| `ultralytics` | YOLOv11 model inference |
| `opencv-python-headless` | Image and video processing |
| `numpy` | Array operations |
| `Pillow` | Image format handling |
| `torch` | PyTorch inference engine |
| `torchvision` | Vision utilities |

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
