# 💧 Glacial Pathfinder (Watermark Remover)

A high-performance AI watermark removal tool built with Streamlit and LaMa (Large Mask Inpainting).

## Features
- **AI-Powered**: Uses Big-LaMa model for high-quality inpainting.
- **Tiled Inference**: Supports high-resolution images (2K/4K) without memory overflow.
- **User Friendly**: Simple web interface to draw masks and remove watermarks.
- **Privacy First**: Everything runs locally; no images are sent to the cloud.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cervelo874-dev/glacial-pathfinder.git
   cd glacial-pathfinder
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires Python 3.8+*

3. **Download Model**
   Place the `big-lama.pt` model file in the `models/` directory.
   (You can download it from [HuggingFace](https://huggingface.co/smartywu/big-lama) or other sources).

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

1. Upload an image.
2. Draw over the watermark using the canvas tool.
3. Click "Remove Watermark".
4. Download the result.

## License
MIT License
