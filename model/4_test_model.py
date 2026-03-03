"""
Step 4: Run inference on validation images and display results.

Usage:
    python 4_test_model.py
    python 4_test_model.py --weights runs/detect/train/weights/best.pt --source data/validation/images
    python 4_test_model.py --conf 0.4 --show 20
"""

import argparse
import glob
from pathlib import Path

from ultralytics import YOLO


def run_predict(
    weights: str = "runs/detect/train/weights/best.pt",
    source:  str = "data/validation/images",
    conf:    float = 0.25,
) -> Path:
    """Run YOLO prediction and save results with bounding boxes + label txt files."""
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Run step 3 (train_model.py) first."
        )

    print(f"Running inference with {weights} on {source}")
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=True,       # save annotated images
        save_txt=True,   # save label txt files (needed for step 5)
    )

    predict_dir = Path(results[0].save_dir)
    print(f"Results saved to: {predict_dir}")
    return predict_dir


def display_results(predict_dir: Path, max_images: int = 10) -> None:
    """Display predicted images inline (works in Jupyter / IPython)."""
    try:
        from IPython.display import Image, display
        in_ipython = True
    except ImportError:
        in_ipython = False

    images = sorted(predict_dir.glob("*.jpg"))[:max_images]
    if not images:
        images = sorted(predict_dir.glob("*.png"))[:max_images]

    if not images:
        print(f"[INFO] No result images found in {predict_dir}")
        return

    print(f"\nShowing {len(images)} prediction result(s):")
    for img_path in images:
        print(f"  {img_path}")
        if in_ipython:
            display(Image(filename=str(img_path), height=400))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on validation images.")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    parser.add_argument("--source",  default="data/validation/images")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show",    type=int,   default=10,   help="Max images to display")
    args = parser.parse_args()

    predict_dir = run_predict(args.weights, args.source, args.conf)
    display_results(predict_dir, args.show)
