# SchedBuddy-ML

**Project Overview**
- **Purpose:**: Schedule builder that integrates artificial intelligence to automatically generate timetable, specifically for Bicol University students.
-**Machine Learning resources:** Label Studio, YOLOv8
---
**Training the Model**
1. Upload the dataset. With Label Studio, the folder structure should be:
```
├──images # folder containing the images
├──labels # folder containing the labels in YOLO annotation format
├──classes.txt # labelmap file that contains all the classes
```

2. Split images into train and validation data
```
cd label-studio
python train_val_split.py --datapath="[Path to dataset]" --train_pct=0.7
```

3. Install requirements (Ultralytics)
```
pip install ultralytics
```

4.  Create Ultralytics training configuration YAML file
```
cd label-studio
python create-yaml.py
```

5. Train model
- **Model architecture & size (model):** Larger models run slower but have higher accuracy, while smaller models run faster but have lower accuracy. You can train YOLOv8 or YOLOv5 models by substituting yolo11 for yolov8 or yolov5.
- **Number of epochs (epochs):** In machine learning, one “epoch” is one single pass through the full training dataset. If your dataset has less than 200 images, a good starting point is 60 epochs. If your dataset has more than 200 images, a good starting point is 40 epochs.
- **Resolution (imgsz):** Resolution has a large impact on the speed and accuracy of the model: a lower resolution model will have higher speed but less accuracy. YOLO models are typically trained and inferenced at a 640x640 resolution. However, if you want your model to run faster or know you will be working with low-resolution images, try using a lower resolution like 480x480.
```
yolo detect train data=/content/data.yaml model=yolov8.pt epochs=60 imgsz=640
```

6. Test Model
```
yolo detect predict model=runs/detect/train/weights/best.pt source=data/validation/images save=True

cd label-studio
python test-model.py
```
You can check the results at `runs/detect/predict` folder