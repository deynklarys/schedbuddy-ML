import cv2
from pathlib import Path

IMAGE_FOLDER = Path("runs/detect/predict")
LABEL_FOLDER = IMAGE_FOLDER / "labels"
OUTPUT_FOLDER = Path("cropped_tables")
TABLE_CLASS_ID = 0


def main() -> None:
    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")
    if not LABEL_FOLDER.exists():
        raise FileNotFoundError(
            f"Label folder not found: {LABEL_FOLDER}. "
            "Run predict with save_txt=True to create exact box coordinates."
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_files = [
        path for path in IMAGE_FOLDER.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        label_path = LABEL_FOLDER / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipped (no label file): {label_path}")
            continue

        height, width = image.shape[:2]
        with label_path.open("r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        table_count = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))
            if cls_id != TABLE_CLASS_ID:
                continue

            x_center_n, y_center_n, box_width_n, box_height_n = map(float, parts[1:5])

            box_width = box_width_n * width
            box_height = box_height_n * height
            x_center = x_center_n * width
            y_center = y_center_n * height

            x1 = int(round(x_center - box_width / 2))
            y1 = int(round(y_center - box_height / 2))
            x2 = int(round(x_center + box_width / 2))
            y2 = int(round(y_center + box_height / 2))

            x1 = max(0, min(x1, image.shape[1]))
            x2 = max(0, min(x2, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            y2 = max(0, min(y2, image.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            table_count += 1
            output_path = OUTPUT_FOLDER / f"{image_path.stem}_table_{table_count}{image_path.suffix}"
            cv2.imwrite(str(output_path), cropped)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()