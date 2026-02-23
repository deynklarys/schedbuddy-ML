#!/usr/bin/env python3
"""
YOLO Model Training Script
Trains a YOLO model for object detection
"""

from ultralytics import YOLO

def train_model(
    data_yaml: str = "data.yaml",
    model: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640
):
    """
    Train YOLO model
    
    Args:
        data_yaml: Path to the data configuration YAML file
        model: Model architecture to use (e.g., yolov8.pt, yolov11s.pt)
        epochs: Number of training epochs
        imgsz: Image resolution for training
    """
    print(f"Starting YOLO training...")
    print(f"  Data YAML: {data_yaml}")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {imgsz}x{imgsz}")
    
    # Load model
    model = YOLO(model)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=-1  # GPU device, set to 0 for first GPU, or -1 for CPU
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    train_model(
        data_yaml="data.yaml",
        model="yolo11s.pt",
        epochs=60,
        imgsz=640
    )
