import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import requests

from streamlit_drawable_canvas import st_canvas
from core.utils import load_image, dilute_mask
from core.inpainter import WatermarkRemover

# --- Configuration ---
MODEL_URL = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
MODEL_PATH = "models/big-lama.pt"

st.set_page_config(layout="wide", page_title="Watermark Remover AI")

# --- Utils ---
def ensure_model_ready():
    """Downloads model if missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        st.info("📥 Downloading AI Model (approx. 200MB)... This happens only once.")
        try:
            with st.spinner("Downloading..."):
                resp = requests.get(MODEL_URL, stream=True)
                resp.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
            st.success("✅ Model Ready!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    elif os.path.getsize(MODEL_PATH) < 1000:
        os.remove(MODEL_PATH)
        st.experimental_rerun()

# --- Main App ---
ensure_model_ready()

st.title("💧 AI Watermark Remover")

# Session State Initialization
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "file_id" not in st.session_state:
    st.session_state.file_id = None

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    stroke_width = st.slider("Brush Size", 5, 50, 20)
    dilation = st.slider("Mask Expansion", 0, 10, 2)
    st.info("Use the canvas to draw over watermarks.")

# Main Layout
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    # Load logic: Only process if a new file is uploaded
    file_id = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.file_id != file_id:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.original_image = image
            st.session_state.processed_image = None
            st.session_state.file_id = file_id
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Display Canvas
    if st.session_state.original_image:
        original = st.session_state.original_image
        w, h = original.size
        
        # Responsive Resizing
        display_width = 700
        canvas_width = min(w, display_width)
        scale = canvas_width / w
        canvas_height = int(h * scale)
        
        # Prepare background for canvas
        bg_image = original.resize((canvas_width, canvas_height))
        
        # Canvas
        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1.0)",
                stroke_width=stroke_width,
                stroke_color="#fff",
                background_image=bg_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                key=f"canvas_{file_id}", # Unique key per file
            )
        except Exception as e:
            st.error(f"Canvas Error: {e}")
            st.stop()

        if st.button("✨ Remove Watermark", type="primary"):
            if canvas_result.image_data is not None:
                with st.spinner("Removing watermark..."):
                    # Get Alpha channel from canvas
                    mask_data = canvas_result.image_data[:, :, 3] 
                    mask_pil = Image.fromarray(mask_data).resize((w, h)) # Resize mask to match original
                    
                    # Dilation
                    mask_dilated = dilute_mask(mask_pil, iterations=dilation)
                    
                    # Inference
                    try:
                        remover = WatermarkRemover(model_path=MODEL_PATH)
                        # Simply calling process, ignoring tiling for simplicity/robustness first
                        result = remover.process_image(original, mask_dilated, use_tiling=True)
                        st.session_state.processed_image = result
                    except Exception as e:
                        st.error(f"Processing Error: {e}")

# Results Display
if st.session_state.processed_image:
    st.subheader("2. Result")
    c1, c2 = st.columns(2)
    c1.image(st.session_state.original_image, caption="Original", use_column_width=True)
    c2.image(st.session_state.processed_image, caption="Cleaned", use_column_width=True)
    
    # Download
    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Result", 
        data=buf.getvalue(), 
        file_name="cleaned_image.png", 
        mime="image/png"
    )
