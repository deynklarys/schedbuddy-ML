from ultralytics import YOLO

# Use pre-trained document layout model
model = YOLO('yolov8n.pt')  # Start with nano (fast)

# Or use a document-specific model
# model = YOLO('publaynet_yolov8.pt')  # Pre-trained on documents

results = model('images/original/watermarked.png')
results[0].show()  # Visualize detections