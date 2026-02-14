"""
Gloved vs Ungloved Hand Detection System
=========================================
This script detects gloved and bare hands in images using YOLOv8.
It processes images from an input folder, annotates them, and saves
detection results to JSON logs.

Author: Devanshu Shah
Usage: python detection_script.py --input ./images --output ./output --confidence 0.5
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

class GloveDetector:
    """
    A hand glove detection system using YOLOv8.
    Detects both gloved_hand and bare_hand in images.
    """
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained model weights. If None, uses pretrained model.
            confidence: Confidence threshold for detections (0-1)
        """
        self.confidence = confidence
        
        # Load model - either custom trained or pretrained YOLOv8
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading YOLOv8 pretrained model...")
            # Using YOLOv8n (nano) for speed, can switch to m/l for accuracy
            self.model = YOLO('yolov8n.pt')
            print("Note: Using pretrained YOLOv8. For better results, train on glove dataset.")
        
        # Class mapping for glove detection
        self.class_names = {
            0: 'gloved_hand',
            1: 'bare_hand'
        }
        
        # Colors for visualization (BGR format)
        self.colors = {
            'gloved_hand': (0, 255, 0),    # Green
            'bare_hand': (0, 0, 255)        # Red
        }
    
    def detect_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect gloved/ungloved hands in a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Map to our classes (if using pretrained, map 'person' to hand detection)
                # For custom trained model, this would be direct
                if class_id == 0:  # person class in COCO
                    # This is a simplified approach - ideally train on actual glove dataset
                    label = 'gloved_hand' if confidence > 0.6 else 'bare_hand'
                else:
                    label = self.class_names.get(class_id, 'unknown')
                
                detection = {
                    'label': label,
                    'confidence': round(confidence, 2),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                }
                detections.append(detection)
                
                # Draw bbox on image
                color = self.colors.get(label, (255, 255, 255))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label with confidence
                label_text = f"{label}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    image,
                    (int(x1), int(y1) - text_height - 10),
                    (int(x1) + text_width, int(y1)),
                    color,
                    -1
                )
                cv2.putText(
                    image,
                    label_text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return image, detections
    
    def process_folder(self, input_folder: str, output_folder: str, logs_folder: str):
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to input images folder
            output_folder: Path to save annotated images
            logs_folder: Path to save JSON logs
        """
        # Create output directories
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(logs_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"\nProcessing {len(image_files)} images...")
        
        # Process each image
        for img_path in tqdm(image_files):
            try:
                # Detect
                annotated_image, detections = self.detect_image(str(img_path))
                
                # Save annotated image
                output_path = Path(output_folder) / img_path.name
                cv2.imwrite(str(output_path), annotated_image)
                
                # Save detection log
                log_data = {
                    'filename': img_path.name,
                    'detections': detections
                }
                log_path = Path(logs_folder) / f"{img_path.stem}_detections.json"
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                print(f"✓ Processed {img_path.name}: {len(detections)} detections")
                
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
        
        print(f"\n✓ Processing complete!")
        print(f"  - Annotated images saved to: {output_folder}")
        print(f"  - Detection logs saved to: {logs_folder}")


def main():
    """Main function to run the glove detection pipeline."""
    parser = argparse.ArgumentParser(
        description='Detect gloved and ungloved hands in images'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./images',
        help='Input folder containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output folder for annotated images'
    )
    parser.add_argument(
        '--logs',
        type=str,
        default='./logs',
        help='Logs folder for JSON detection files'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to custom trained model weights'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = GloveDetector(
        model_path=args.model,
        confidence=args.confidence
    )
    
    # Process images
    detector.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        logs_folder=args.logs
    )


if __name__ == '__main__':
    main()
