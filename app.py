import os
import tempfile
import uuid
import pandas as pd
import torch
import streamlit as st
from PIL import Image

# Import utility functions
from utils import load_model, predict_image, predict_video, cleanup_outputs_folder

# ----------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Detection using YOLOv11",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark-accented premium styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Apply typography */
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Metrics panel enhancement */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #ef4444;
    }
    
    /* Title font sizes */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    
    /* Section dividers */
    hr {
        margin: 20px 0;
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Clean up stale temp files in outputs/ folder on startup
try:
    cleanup_outputs_folder("outputs")
except Exception:
    pass

# ----------------------------------------------------
# 2. MODEL INFERENCE PREPARATION
# ----------------------------------------------------
# Load the model and display error gracefully if missing
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Model Loading Failure: {e}")
    st.info("💡 Please make sure 'best.pt' is placed in the project root directory and is a valid PyTorch weights file.")
    st.stop()

# Get classes from the loaded model
detectable_classes = model.names if model_loaded else {}

# ----------------------------------------------------
# 3. SIDEBAR LAYOUT
# ----------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Inference Settings")
    st.markdown("Adjust parameters to tune model sensitivity.")
    
    # Confidence Slider
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required to detect an object."
    )
    
    # IoU Slider
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression (NMS)."
    )
    
    st.markdown("---")
    
    # Model Information Section
    st.markdown("### 📊 Model Information")
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    
    st.info(
        f"**Framework:** Ultralytics YOLOv11\n\n"
        f"**Weights File:** `best.pt`\n\n"
        f"**Running Device:** `{device_name}`\n\n"
        f"**Task:** Object Detection"
    )
    
    # List Detectable Classes
    st.markdown("### 🏷️ Detectable Classes")
    if detectable_classes:
        for cid, name in detectable_classes.items():
            st.markdown(f"- `{cid}`: **{name.capitalize()}**")
    else:
        st.markdown("*No classes found*")

# ----------------------------------------------------
# 4. MAIN PANEL LAYOUT
# ----------------------------------------------------
st.markdown('<div class="main-title">🔥 Forest Fire Detection using YOLOv11</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image or video to detect forest fires.</div>', unsafe_allow_html=True)

# Layout tab pages
tab_image, tab_video = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

# ----------------------------------------------------
# 5. TAB 1: IMAGE DETECTION
# ----------------------------------------------------
with tab_image:
    st.markdown("### Upload Image")
    uploaded_image_file = st.file_uploader(
        "Choose an image file...",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        help="Upload JPG, JPEG, or PNG images."
    )
    
    if uploaded_image_file is not None:
        try:
            # 1. Open the image
            image = Image.open(uploaded_image_file)
            
            # Check if file is corrupted
            image.verify()
            # Re-open because verify() closes the file pointer
            image = Image.open(uploaded_image_file)
            
            # 2. Run prediction
            with st.spinner("Analyzing image..."):
                annotated_img, detections, metrics = predict_image(
                    model=model,
                    image=image,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
            
            st.success("Analysis complete!")
            
            # 3. Side-by-side comparison
            col_orig, col_det = st.columns(2)
            with col_orig:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)
                
            with col_det:
                st.markdown("#### Detected Image")
                st.image(annotated_img, use_container_width=True)
                
            # 4. Metrics Dashboard
            st.markdown("---")
            st.markdown("### 📊 Metrics Dashboard")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Detection Count", f"{metrics['count']} objects")
            with m_col2:
                st.metric("Inference Time", f"{metrics['inference_time']} ms")
            with m_col3:
                st.metric("Average Confidence", f"{metrics['avg_confidence'] * 100:.1f}%" if metrics['count'] > 0 else "0.0%")
            with m_col4:
                st.metric("Highest Confidence", f"{metrics['max_confidence'] * 100:.1f}%" if metrics['count'] > 0 else "0.0%")
            
            # 5. Detections Table & Download Buttons
            st.markdown("---")
            col_table, col_actions = st.columns([2, 1])
            
            with col_table:
                st.markdown("### 📋 Detection Details")
                if len(detections) > 0:
                    df = pd.DataFrame(detections)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No forest fire or smoke detections found with the current confidence threshold.")
            
            with col_actions:
                st.markdown("### 📥 Download Output")
                # Save annotated image bytes
                from io import BytesIO
                img_byte_arr = BytesIO()
                
                # Handle image formats dynamically
                orig_format = uploaded_image_file.type.split('/')[-1].upper()
                if orig_format == "JPG":
                    orig_format = "JPEG"
                elif orig_format not in ["JPEG", "PNG"]:
                    orig_format = "JPEG"
                
                annotated_img.save(img_byte_arr, format=orig_format)
                img_bytes = img_byte_arr.getvalue()
                
                st.download_button(
                    label="Download Annotated Image",
                    data=img_bytes,
                    file_name=f"detected_{uploaded_image_file.name}",
                    mime=uploaded_image_file.type,
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing image file: {e}")
            st.info("💡 Please verify that the file is a valid, uncorrupted image (JPG, JPEG, or PNG).")

# ----------------------------------------------------
# 6. TAB 2: VIDEO DETECTION
# ----------------------------------------------------
with tab_video:
    st.markdown("### Upload Video")
    uploaded_video_file = st.file_uploader(
        "Choose a video file...",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        help="Upload MP4, AVI, or MOV video files."
    )
    
    if uploaded_video_file is not None:
        # Create output directory if not exists
        os.makedirs("outputs", exist_ok=True)
        
        # Define output filename
        out_ext = ".mp4"
        output_filename = f"processed_{uuid.uuid4().hex}{out_ext}"
        output_filepath = os.path.join("outputs", output_filename)
        
        # Write uploaded video stream into a temp file for OpenCV reader
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_in_file:
                temp_in_file.write(uploaded_video_file.read())
                temp_in_path = temp_in_file.name
                
            st.info("Video successfully uploaded! Click 'Run Detection' below to start the inference process.")
            
            # Button to trigger video analysis
            if st.button("🚀 Run Detection", use_container_width=True):
                # Progress placeholders
                st.markdown("---")
                st.markdown("### ⏳ Processing Progress")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                # Callback function to update streamlit components
                def video_progress_callback(progress_ratio, current_frame, total_frames):
                    progress_bar.progress(progress_ratio)
                    status_text.text(
                        f"Processed {current_frame} / {total_frames} frames "
                        f"({int(progress_ratio * 100)}%)"
                    )
                
                processed_video_bytes = None
                video_metrics = None
                
                # Run processing
                try:
                    video_metrics = predict_video(
                        model=model,
                        input_path=temp_in_path,
                        output_path=output_filepath,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        progress_callback=video_progress_callback
                    )
                    
                    # Read output video bytes into memory immediately
                    if os.path.exists(output_filepath):
                        with open(output_filepath, "rb") as f:
                            processed_video_bytes = f.read()
                    
                    st.success("Video processing completed successfully!")
                    
                except Exception as ex:
                    st.error(f"❌ Error during video processing: {ex}")
                    
                finally:
                    # Cleanup input temp file immediately to release disk space
                    if os.path.exists(temp_in_path):
                        try:
                            os.remove(temp_in_path)
                        except Exception:
                            pass
                    # Cleanup output video file from disk immediately
                    if os.path.exists(output_filepath):
                        try:
                            os.remove(output_filepath)
                        except Exception:
                            pass
                
                # If processing succeeded, display results
                if processed_video_bytes is not None and video_metrics is not None:
                    # Display metrics dashboard
                    st.markdown("---")
                    st.markdown("### 📊 Video Metrics Dashboard")
                    
                    vm_col1, vm_col2, vm_col3, vm_col4 = st.columns(4)
                    with vm_col1:
                        st.metric("Total Detections", f"{video_metrics['total_detections']} items")
                    with vm_col2:
                        st.metric("Processing Time", f"{video_metrics['total_time']} seconds")
                    with vm_col3:
                        st.metric("Average Speed", f"{video_metrics['avg_frame_time']} ms/frame")
                    with vm_col4:
                        st.metric("Avg / Max Confidence", f"{video_metrics['avg_confidence'] * 100:.1f}% / {video_metrics['max_confidence'] * 100:.1f}%")
                    
                    # Video actions and download button
                    st.markdown("---")
                    col_player, col_download = st.columns([2, 1])
                    
                    with col_player:
                        st.markdown("### Processed Video Output")
                        try:
                            # Stream the output video directly in Streamlit using the loaded bytes
                            st.video(processed_video_bytes)
                        except Exception as ve:
                            st.warning(f"Unable to play video directly in browser: {ve}")
                            
                    with col_download:
                        st.markdown("### 📥 Download Output")
                        st.download_button(
                            label="Download Processed Video",
                            data=processed_video_bytes,
                            file_name=f"detected_{uploaded_video_file.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
                        st.markdown("")
                        st.info(
                            "💡 **Browser Playback Note:** Some browsers do not support online playback "
                            "for specific video codecs. If you get a black screen or playback failure, "
                            "please use the button above to download the video and play it in a media player (like VLC)."
                        )
            
        except Exception as e:
            st.error(f"❌ Error initializing video file: {e}")
            st.info("💡 Please verify that the file is a valid, uncorrupted video (MP4, AVI, or MOV).")
