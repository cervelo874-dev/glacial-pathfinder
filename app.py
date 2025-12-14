import streamlit as st
from PIL import Image
import numpy as np
import io
import os

from core.utils import load_image, dilute_mask, resize_image_max_edge
from core.inpainter import WatermarkRemover

from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(layout="wide", page_title="Watermark Remover AI")

# Initialize Session State
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar
st.sidebar.title("Watermark Remover")
st.sidebar.markdown("### Settings")
stroke_width = st.sidebar.slider("Brush Width", 1, 50, 15)
dilation_iterations = st.sidebar.slider("Mask Dilation", 0, 10, 2)
use_tiling = st.sidebar.checkbox("Use Tiling (for large images)", value=True)
tile_size = st.sidebar.number_input("Tile Size", value=1024, step=128)

st.sidebar.markdown("---")
st.sidebar.info("Model: LaMa (Big-LaMa)")

# Main Interface
st.title("💧 AI Watermark Remover")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None or st.session_state.original_image is not None:
    # Handle File Load
    try:
        if uploaded_file is not None and (st.session_state.current_file != uploaded_file.name):
            image = load_image(uploaded_file)
            st.session_state.original_image = image
            st.session_state.current_file = uploaded_file.name
            st.session_state.processed_image = None # Reset
            st.rerun()
    except Exception as e:
        st.error(f"Error loading image: {e}")
            
    original_image = st.session_state.original_image
    
    if original_image:
        w, h = original_image.size
        
        st.write(f"Original Resolution: {w}x{h}")

        # Canvas for Masking
        st.subheader("1. Mark the Watermark")
        st.markdown("Draw over the watermark/object you want to remove.")
        
        display_width = 700
        if w > display_width:
            scale_factor = display_width / w
            canvas_width = display_width
            canvas_height = int(h * scale_factor)
            display_image = original_image.resize((canvas_width, canvas_height))
        else:
            scale_factor = 1.0
            canvas_width = w
            canvas_height = h
            display_image = original_image

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",  # Drawing with white
            stroke_width=stroke_width,
            stroke_color="#fff",
            background_image=display_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.current_file}", # Check reuse
        )

        # Process Button
        if st.button("🚀 Remove Watermark"):
            if canvas_result.image_data is not None:
                with st.spinner("Processing... This may take a moment."):
                    # Get user drawn mask (RGBA)
                    mask_data = canvas_result.image_data
                    
                    # This mask is at display resolution. Need to resize to original.
                    mask_pil = Image.fromarray(mask_data.astype('uint8'), mode="RGBA")
                    
                    if scale_factor != 1.0:
                        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                    
                    # Dilation
                    mask_dilated = dilute_mask(mask_pil, iterations=dilation_iterations)
                    
                    # Inpaint
                    remover = WatermarkRemover(model_path="models/big-lama.pt")
                    try:
                        result = remover.process_image(
                            original_image, 
                            mask_dilated, 
                            use_tiling=use_tiling,
                            tile_size=tile_size
                        )
                        st.session_state.processed_image = result
                        st.success("Removal Complete!")
                    except FileNotFoundError as e:
                        st.error(str(e))
                        st.info("💡 You need to place 'big-lama.pt' in the 'models/' directory.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                 st.warning("Please draw on the image to select the area to remove.")

        # Results
        if st.session_state.processed_image is not None:
             st.subheader("2. Result")
             
             col1, col2 = st.columns(2)
             with col1:
                 st.image(st.session_state.original_image, caption="Original", use_column_width=True)
             with col2:
                 st.image(st.session_state.processed_image, caption="Processed", use_column_width=True)
             
             # Download
             buf = io.BytesIO()
             st.session_state.processed_image.save(buf, format="PNG")
             byte_im = buf.getvalue()
             
             st.download_button(
                 label="Download Clean Image (PNG)",
                 data=byte_im,
                 file_name="cleaned_image.png",
                 mime="image/png"
             )

else:
    st.info("Please upload an image to start.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using LaMa and Streamlit")
