import os
import torch
import numpy as np
from PIL import Image
import threading

from .tiler import Tiler
from .utils import pil_to_cv2, cv2_to_pil, resize_image_max_edge

class WatermarkRemover:
    def __init__(self, model_path="models/big-lama.pt", device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self._lock = threading.Lock()
        
    def load_model(self):
        """Loads the TorchScript LaMa model."""
        if self.model is not None:
            return

        with self._lock:
            if self.model is not None:
                return
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}. Please download 'big-lama.pt' and place it in the models directory.")
            
            try:
                print(f"Loading model from {self.model_path} to {self.device}...")
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
                print("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

    def _predict(self, image: np.ndarray, mask: np.ndarray):
        """
        Runs the actual inference on a single image/mask pair (H, W, C).
        Image and Mask should be numpy arrays (0-255).
        Mask is single channel.
        """
        # Preprocess
        # LaMa expects [0,1] float inputs, (B, C, H, W)
        
        img_t = torch.from_numpy(image).float().div(255.0)
        mask_t = torch.from_numpy(mask).float().div(255.0)
        
        # Add batch dim and channel dim to mask if needed
        img_t = img_t.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        
        if len(mask_t.shape) == 2:
            mask_t = mask_t.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        elif len(mask_t.shape) == 3:
            mask_t = mask_t.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W) -> usually 1 channel
        
        # Verify divisibility by 8 (required for some models like LaMa, usually needs mod 8 or mod 32)
        # We assume Tiler handles this or we pad here.
        # But for arbitrary inference, we should pad.
        h, w = img_t.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
            mask_t = torch.nn.functional.pad(mask_t, (0, pad_w, 0, pad_h), mode='reflect')
            
        img_t = img_t.to(self.device)
        mask_t = mask_t.to(self.device)
        
        with torch.no_grad():
            # LaMa input signature usually: image, mask
            # But big-lama.pt might be traced with specific expectations.
            # Usually it expects: forward(image, mask)
            output = self.model(img_t, mask_t)
            
            # If output is a tuple (common in some repos), take first
            if isinstance(output, (tuple, list)):
                output = output[0]
        
        # Postprocess
        output = output.cpu()
        
        # Crop padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]
            
        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255
        return output.astype(np.uint8)

    def process_image(self, image_pil: Image.Image, mask_pil: Image.Image, use_tiling=True, tile_size=1024):
        """
        Main entry point.
        image_pil: RGB Pillow Image
        mask_pil: L or RGB Pillow Image (white = mask)
        """
        self.load_model()
        
        # Convert to numpy
        img_np = np.array(image_pil)
        mask_np = np.array(mask_pil.convert("L"))
        
        # Check size
        h, w = img_np.shape[:2]
        
        # Decide tiling
        if use_tiling and (h > tile_size or w > tile_size):
            print(f"Image size {w}x{h} > {tile_size}, using Tiling...")
            tiler = Tiler(tile_size=tile_size, overlap=64) # Overlap is fixed or configurable
            
            # Split
            image_tiles = tiler.split(img_np)
            mask_tiles = tiler.split(mask_np[:, :, np.newaxis]) # split expects H,W,C
            
            processed_tiles = []
            
            for (img_tile, y, x, _, _), (mask_tile, _, _, _, _) in zip(image_tiles, mask_tiles):
                # Ensure mask is 2D for processing logic/debugging logic but _predict handles it
                # mask_tile is (H, W, 1) from split
                
                # Run inference on tile
                # If mask is empty, skip inference? No, context might change? 
                # Actually, if mask is all black (0), output == input.
                if np.max(mask_tile) == 0:
                    out_tile = img_tile
                else:
                    out_tile = self._predict(img_tile, mask_tile[:, :, 0])
                
                # Append result
                # Tiler.merge expects (tile, y, x, ph, pw)
                # But we constructed it manually. 
                # Let's match the structure expected by Tiler.merge (which I defined as accepting the tuple list)
                processed_tiles.append((out_tile, y, x, img_tile.shape[0], img_tile.shape[1]))
            
            # Merge
            result_np = tiler.merge(processed_tiles, img_np.shape)
            return Image.fromarray(result_np)
            
        else:
            # Direct inference
            result_np = self._predict(img_np, mask_np)
            return Image.fromarray(result_np)
