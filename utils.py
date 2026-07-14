import os
import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch

# Dynamically set device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    """
    Loads and caches the YOLOv11 model to prevent reloading on each run.
    """
    model_path = "best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please ensure 'best.pt' is in the project root directory."
        )
    try:
        return YOLO("best.pt")
    except Exception as e:
        raise RuntimeError(f"Error initializing YOLOv11 model: {e}")

def predict_image(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    Runs YOLOv11 inference on a single image.
    
    Args:
        model: Loaded YOLO model.
        image (PIL.Image or np.ndarray): Input image.
        conf_threshold (float): Confidence threshold for predictions.
        iou_threshold (float): Intersection over Union (IoU) threshold for NMS.
        
    Returns:
        annotated_img (PIL.Image): Image annotated with bounding boxes and labels.
        detections (list of dicts): Details of detected objects (number, class, confidence, bounding box).
        metrics (dict): Dictionary with summary metrics (count, inference_time, avg_confidence, max_confidence).
    """
    # Perform inference
    results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)
    result = results[0]
    
    # Extract prediction speed (ms)
    speed = result.speed
    preprocess_speed = speed.get('preprocess', 0.0)
    inference_speed = speed.get('inference', 0.0)
    postprocess_speed = speed.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed
    
    # Get annotated image (BGR format) and convert to PIL Image (RGB)
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated_rgb)
    
    # Parse detections
    detections = []
    boxes = result.boxes
    confidences = []
    
    for idx, box in enumerate(boxes):
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Bounding box coordinates: [xmin, ymin, xmax, ymax]
        xyxy = box.xyxy[0].tolist()
        bbox_str = f"[{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]"
        
        detections.append({
            "Detection Number": idx + 1,
            "Class Name": class_name,
            "Confidence": round(conf, 4),
            "Bounding Box": bbox_str
        })
        confidences.append(conf)
        
    metrics = {
        "count": len(detections),
        "inference_time": round(total_speed, 2),  # in ms
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0.0
    }
    
    return annotated_img, detections, metrics

def predict_video(model, input_path, output_path, conf_threshold=0.25, iou_threshold=0.45, progress_callback=None):
    """
    Runs frame-by-frame YOLOv11 inference on an uploaded video and writes the annotated video to disk.
    
    Args:
        model: Loaded YOLO model.
        input_path (str): Path to temporary uploaded video.
        output_path (str): Destination path inside outputs/ for the processed video.
        conf_threshold (float): Confidence threshold.
        iou_threshold (float): IoU threshold.
        progress_callback (callable, optional): Function accepting (progress_ratio, current_frame, total_frames).
        
    Returns:
        metrics (dict): Aggregated detection statistics.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try using 'avc1' first (H.264) for HTML5 video compatibility. If it fails, fallback to 'mp4v'.
    out = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            out.release()
            out = None
    except Exception:
        out = None
        
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    if not out.isOpened():
        cap.release()
        raise IOError(f"Could not open video writer with output path: {output_path}")
        
    frame_count = 0
    total_detections = 0
    all_confidences = []
    total_inference_time = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Predict frame-by-frame. OpenCV frame format is BGR.
            results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)
            result = results[0]
            
            # Draw predictions on frame
            annotated_frame = result.plot()
            
            # Write annotated frame
            out.write(annotated_frame)
            
            end_time = time.time()
            frame_time_ms = (end_time - start_time) * 1000.0
            total_inference_time += frame_time_ms
            
            # Accumulate detection counts and confidence scores
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                all_confidences.append(conf)
                total_detections += 1
                
            frame_count += 1
            
            # Call the progress update callback if supplied
            if progress_callback and total_frames > 0:
                progress_ratio = min(frame_count / total_frames, 1.0)
                progress_callback(progress_ratio, frame_count, total_frames)
                
    finally:
        cap.release()
        out.release()
        
    avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
    max_conf = float(np.max(all_confidences)) if all_confidences else 0.0
    avg_frame_time = (total_inference_time / frame_count) if frame_count > 0 else 0.0
    
    metrics = {
        "total_frames": frame_count,
        "total_detections": total_detections,
        "avg_confidence": round(avg_conf, 4),
        "max_confidence": round(max_conf, 4),
        "avg_frame_time": round(avg_frame_time, 2),  # ms
        "total_time": round(total_inference_time / 1000.0, 2)  # seconds
    }
    
    return metrics

def cleanup_outputs_folder(folder_path="outputs", max_age_seconds=1800):
    """
    Scans the outputs directory and deletes any generated video/image files older than 30 minutes.
    This prevents running out of disk space on remote Streamlit servers.
    """
    if not os.path.exists(folder_path):
        return
    now = time.time()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename == ".gitkeep":
            continue
        try:
            if os.path.isfile(file_path):
                if os.stat(file_path).st_mtime < now - max_age_seconds:
                    os.remove(file_path)
        except Exception as e:
            # Silent catch or print to logs
            pass
