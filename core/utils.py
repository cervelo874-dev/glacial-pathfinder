import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_file):
    """
    Loads an image from a file-like object (e.g. Streamlit uploader)
    and returns a PIL Image.
    """
    return Image.open(image_file).convert("RGB")

def pil_to_cv2(pil_image):
    """Converts a PIL image to a standard OpenCV (BGR) format numpy array."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Converts an OpenCV image (BGR) to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def dilute_mask(mask_pil, kernel_size=5, iterations=2):
    """
    Dilates a binary mask to ensure the watermark is fully covered.
    Input: PIL Image (L mode preferred)
    Output: PIL Image
    """
    mask_np = np.array(mask_pil.convert("L"))
    
    # Ensure binary (0 or 255)
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=iterations)
    
    return Image.fromarray(dilated)

def resize_image_max_edge(image: Image.Image, max_edge: int = 2048) -> Image.Image:
    """
    Resizes image so that the longest edge is at most max_edge.
    """
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    
    scale = max_edge / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)
