# resize the image
> such that the width is at most 1024 to reduce the image loading speed

## functionality
This script is used for batch resizing images. If the image width is larger than the specified value (default 1024 pixels), it will be resized to the specified width while maintaining the aspect ratio.

## Usage

### Basic Usage
```bash
# Resize all images in /raid/lilong/data/face_seg/ directory
python script/resize_images.py /raid/lilong/data/face_seg/

# Specify output directory
python script/resize_images.py /raid/lilong/data/face_seg/ -o /path/to/output/

# Custom maximum width
python script/resize_images.py /raid/lilong/data/face_seg/ -w 800

# In-place modification (overwrite original files)
python script/resize_images.py /raid/lilong/data/face_seg/ --in-place
```

### Parameters
- `input_dir`: Input image directory path (required)
- `-o, --output_dir`: Output directory path (optional, defaults to input_directory_name_resized)
- `-w, --max_width`: Maximum width (optional, default 1024)
- `--in-place`: In-place modification, overwrite original files (use with caution)

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

## Notes
1. The script maintains the original image aspect ratio
2. Using the `--in-place` parameter will overwrite original files, use with caution
3. The script automatically creates output directory structure, maintaining the same hierarchy as input directory
4. Only images with width larger than the specified value will be resized, others remain unchanged
