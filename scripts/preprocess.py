#!/usr/bin/env python3
"""
Preprocess rendered images: segment, grayscale, edge enhance, normalize.

Usage:
    python scripts/preprocess.py --input data/renders --output data/processed
    python scripts/preprocess.py --input image.png --output processed.png
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# === CONFIGURATION ===

TARGET_WIDTH = 512
TARGET_HEIGHT = 682  # 3:4 portrait ratio

# Edge enhancement parameters
SOBEL_KSIZE = 3
EDGE_BLEND_ALPHA = 0.7  # How much of the edge to blend (0=no edge, 1=edges only)


def load_image(path: str) -> np.ndarray:
    """Load image as RGBA numpy array."""
    img = Image.open(path).convert('RGBA')
    return np.array(img)


def extract_mask(img_rgba: np.ndarray) -> np.ndarray:
    """Extract binary mask from alpha channel."""
    alpha = img_rgba[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to image (set background to white)."""
    result = img.copy()
    if len(result.shape) == 2:
        result[mask == 0] = 255  # White background for grayscale
    else:
        result[mask == 0] = [255, 255, 255]  # White background for RGB
    return result


def to_grayscale(img_rgba: np.ndarray) -> np.ndarray:
    """Convert RGBA image to grayscale."""
    # Use standard luminance weights
    rgb = img_rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray


def enhance_edges(gray: np.ndarray, alpha: float = EDGE_BLEND_ALPHA) -> np.ndarray:
    """
    Enhance edges using Sobel filter and blend with original.
    
    Args:
        gray: Grayscale image
        alpha: Blend factor (0 = original, 1 = edges only)
    
    Returns:
        Edge-enhanced grayscale image
    """
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
    
    # Gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    # Invert so edges are dark (like a drawing)
    edges = 255 - magnitude
    
    # Blend with original grayscale
    blended = cv2.addWeighted(
        gray, 1 - alpha,
        edges, alpha,
        0
    )
    
    return blended.astype(np.uint8)


def normalize_scale(img: np.ndarray, mask: np.ndarray, 
                    target_width: int = TARGET_WIDTH,
                    target_height: int = TARGET_HEIGHT) -> tuple:
    """
    Normalize image scale so the body fills a consistent portion of the frame.
    
    Returns:
        Tuple of (normalized_image, normalized_mask)
    """
    # Find bounding box of the body (non-zero mask pixels)
    coords = cv2.findNonZero(mask)
    if coords is None:
        # No body found, return as-is
        return img, mask
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop to bounding box with padding
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    img_cropped = img[y1:y2, x1:x2]
    mask_cropped = mask[y1:y2, x1:x2]
    
    # Calculate scale to fit in target size while maintaining aspect ratio
    crop_h, crop_w = img_cropped.shape[:2]
    scale_w = target_width / crop_w
    scale_h = target_height / crop_h
    scale = min(scale_w, scale_h) * 0.9  # 90% of frame
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    
    # Resize
    img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create canvas and center the body
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    mask_canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    
    # Center position
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    # Paste onto canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized
    
    return canvas, mask_canvas


def preprocess_image(input_path: str, output_path: str) -> dict:
    """
    Full preprocessing pipeline for a single image.
    
    Args:
        input_path: Path to input RGBA image
        output_path: Path to save processed grayscale image
    
    Returns:
        Dict with processing info
    """
    # Load image
    img_rgba = load_image(input_path)
    original_shape = img_rgba.shape[:2]
    
    # Extract mask from alpha channel
    mask = extract_mask(img_rgba)
    
    # Convert to grayscale
    gray = to_grayscale(img_rgba)
    
    # Apply mask (white background)
    gray_masked = apply_mask(gray, mask)
    
    # Enhance edges
    enhanced = enhance_edges(gray_masked)
    
    # Normalize scale and center
    normalized, norm_mask = normalize_scale(enhanced, mask)
    
    # Apply mask again after scaling (clean up edges)
    final = apply_mask(normalized, norm_mask)
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, final)
    
    return {
        "input": input_path,
        "output": output_path,
        "original_shape": original_shape,
        "final_shape": final.shape,
    }


def process_directory(input_dir: str, output_dir: str, pattern: str = "*_front.png") -> list:
    """
    Process all matching images in a directory.
    
    Args:
        input_dir: Directory containing rendered images
        output_dir: Directory to save processed images
        pattern: Glob pattern to match input files
    
    Returns:
        List of processing results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process both front and side views
    for view_pattern in ["*_front.png", "*_side.png"]:
        for img_file in input_path.glob(view_pattern):
            # Generate output filename
            out_file = output_path / img_file.name
            
            try:
                result = preprocess_image(str(img_file), str(out_file))
                result["status"] = "success"
                print(f"✓ Processed: {img_file.name}")
            except Exception as e:
                result = {
                    "input": str(img_file),
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ Error processing {img_file.name}: {e}")
            
            results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess body images")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output image or directory")
    parser.add_argument("--edge-blend", "-e", type=float, default=EDGE_BLEND_ALPHA,
                        help="Edge enhancement blend factor (0-1)")
    
    args = parser.parse_args()
    
    global EDGE_BLEND_ALPHA
    EDGE_BLEND_ALPHA = args.edge_blend
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        result = preprocess_image(args.input, args.output)
        print(f"Processed: {result['input']} -> {result['output']}")
    elif input_path.is_dir():
        # Directory
        results = process_directory(args.input, args.output)
        success = sum(1 for r in results if r.get("status") == "success")
        print(f"\nProcessed {success}/{len(results)} images")
    else:
        print(f"Error: {args.input} is not a file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
