import numpy as np
import cv2

class Tiler:
    def __init__(self, tile_size=512, overlap=64):
        self.tile_size = tile_size
        self.overlap = overlap

    def split(self, image_np):
        """
        Splits an image (H, W, C) into tiles.
        Returns a list of (tile, y, x) tuples.
        """
        h, w, c = image_np.shape
        tiles = []
        
        # Calculate steps
        stride = self.tile_size - self.overlap
        
        y_steps = list(range(0, h, stride))
        x_steps = list(range(0, w, stride))
        
        # Adjust last step to fit exactly if needed, or pad? 
        # Better strategy: if last tile goes out of bounds, shift it back.
        
        final_y_steps = []
        for y in y_steps:
            if y + self.tile_size > h:
                y = max(0, h - self.tile_size)
                final_y_steps.append(y)
                # If we shifted back, we might cover the same area, but that's fine.
                # However, cleaner logic is to just ensure we cover everything.
                # If h < tile_size, we just take 0.
                break
            final_y_steps.append(y)
        if not final_y_steps: # image smaller than tile
            final_y_steps = [0]
        elif final_y_steps[-1] + self.tile_size < h:
             final_y_steps.append(max(0, h - self.tile_size))
             
        final_x_steps = []
        for x in x_steps:
            if x + self.tile_size > w:
                x = max(0, w - self.tile_size)
                final_x_steps.append(x)
                break
            final_x_steps.append(x)
        if not final_x_steps:
             final_x_steps = [0]
        elif final_x_steps[-1] + self.tile_size < w:
             final_x_steps.append(max(0, w - self.tile_size))

        # Unique sort
        final_y_steps = sorted(list(set(final_y_steps)))
        final_x_steps = sorted(list(set(final_x_steps)))

        for y in final_y_steps:
            for x in final_x_steps:
                # Extract tile
                # Handle padding if image is smaller than tile_size
                tile = image_np[y:y+self.tile_size, x:x+self.tile_size]
                
                # Check if we need padding (only if image < tile_size)
                ph, pw, _ = tile.shape
                if ph < self.tile_size or pw < self.tile_size:
                    pad_b = self.tile_size - ph
                    pad_r = self.tile_size - pw
                    tile = np.pad(tile, ((0, pad_b), (0, pad_r), (0, 0)), mode='reflect')
                
                tiles.append((tile, y, x, ph, pw)) # Store original h/w for unpadding if needed
        
        return tiles

    def merge(self, tiles, original_shape):
        """
        Merges tiles back into a single image with blending.
        tiles: list of (tile, y, x, ph, pw) - though processed tiles might not have ph, pw preserved in the same object structure usually.
        We expect 'tiles' input here to be the PROCESSED tiles (numpy arrays), 
        and we need the coordinates.
        So this function signature might need adjustment. 
        Let's assume tiles is a list of (processed_tile_np, y, x, valid_h, valid_w).
        """
        h, w, c = original_shape
        canvas = np.zeros((h, w, c), dtype=np.float32)
        weights = np.zeros((h, w, c), dtype=np.float32)
        
        # Create a weight mask for blending (linear gradient at edges or gaussian)
        # Linear transparency at edges of the tile
        tile_weight = np.ones((self.tile_size, self.tile_size, c), dtype=np.float32)
        
        # Simple linear blending on edges
        blend_dist = self.overlap // 2
        
        # Create 1D gradients
        grad = np.linspace(0, 1, blend_dist)
        # Top
        for i in range(blend_dist):
            tile_weight[i, :, :] *= grad[i]
        # Bottom
        for i in range(blend_dist):
            tile_weight[self.tile_size - 1 - i, :, :] *= grad[i]
        # Left
        for i in range(blend_dist):
            tile_weight[:, i, :] *= grad[i]
        # Right
        for i in range(blend_dist):
            tile_weight[:, self.tile_size - 1 - i, :] *= grad[i]
            
        
        for tile, y, x, vh, vw in tiles:
            # If the tile was padded (image < tile_size), crop it back
            # But the 'tile' here is the output of the model, which should be tile_size x tile_size.
            # However, we only care about the valid region [0:vh, 0:vw] for the final canvas technically?
            # Actually, if we padded with reflection, the model inpainted the reflection too.
            # We should probably just use the full tile and accumulation will handle it, 
            # BUT for the edges of the generic image, we shouldn't blend with non-existent pixels.
            
            # Simple approach: add weighted tile to canvas
            cur_tile_weight = tile_weight.copy()
            
            # If this is an edge tile, we shouldn't fade out the edge that is on the image boundary.
            # Top edge: y == 0
            if y == 0:
                cur_tile_weight[0:blend_dist, :, :] = 1.0
            # Bottom edge: y + self.tile_size >= h (approx check)
            if y + self.tile_size >= h:
                cur_tile_weight[self.tile_size-blend_dist:, :, :] = 1.0
            # Left edge: x == 0
            if x == 0:
                cur_tile_weight[:, 0:blend_dist, :] = 1.0
            # Right edge
            if x + self.tile_size >= w:
                cur_tile_weight[:, self.tile_size-blend_dist:, :] = 1.0
            
            # Crop valid area if image was smaller than tile
            # (If image < tile_size)
            curr_h, curr_w, _ = tile.shape 
            # Note: tile.shape is physically (512, 512) usually.
            
            # Place on canvas
            # Handle boundary conditions precisely
            # Actual placement coordinates
            y_end = min(y + self.tile_size, h)
            x_end = min(x + self.tile_size, w)
            
            h_slice = y_end - y
            w_slice = x_end - x
            
            # Slice the tile and weight to match the placement area (handling the bottom-right crops if any)
            tile_slice = tile[0:h_slice, 0:w_slice]
            weight_slice = cur_tile_weight[0:h_slice, 0:w_slice]
            
            canvas[y:y_end, x:x_end] += tile_slice * weight_slice
            weights[y:y_end, x:x_end] += weight_slice

        # Normalize
        # Avoid division by zero
        weights[weights < 1e-5] = 1.0 
        result = canvas / weights
        
        return np.clip(result, 0, 255).astype(np.uint8)
