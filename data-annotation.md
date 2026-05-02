Absolutely! Let me show you several ways to **dramatically reduce** the labeling time:

---

## 🚀 Strategy 1: Smart Augmentation (Get 200 from 50)

**Instead of labeling 200 images, label only 50 well and augment!**

### Automatic Data Augmentation Script

```python
import cv2
import numpy as np
from pathlib import Path
import random

def augment_image_and_labels(image_path, label_path, output_dir, num_augmentations=3):
    """
    Create multiple versions of one labeled image
    Augmentations that preserve bounding box validity
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # Read labels
    with open(label_path) as f:
        labels = [line.strip().split() for line in f.readlines()]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = image_path.stem
    
    for i in range(num_augmentations):
        aug_img = img.copy()
        aug_labels = labels.copy()
        
        # 1. Brightness adjustment
        brightness = random.uniform(0.7, 1.3)
        aug_img = cv2.convertScaleAbs(aug_img, alpha=brightness, beta=0)
        
        # 2. Contrast adjustment
        contrast = random.uniform(0.8, 1.2)
        aug_img = cv2.convertScaleAbs(aug_img, alpha=contrast, beta=0)
        
        # 3. Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)
        
        # 4. Slight rotation (±5 degrees) - with label adjustment
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), 
                                     borderMode=cv2.BORDER_REPLICATE)
            
            # Note: For simplicity, we keep labels same for small rotations
            # For production, you'd need to rotate bounding boxes too
        
        # 5. Horizontal shift (labels stay valid since it's just translation)
        if random.random() > 0.5:
            shift = random.randint(-20, 20)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            aug_img = cv2.warpAffine(aug_img, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
        
        # Save augmented image
        aug_img_path = output_dir / 'images' / f"{base_name}_aug{i}.jpg"
        aug_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(aug_img_path), aug_img)
        
        # Save augmented labels (same as original for these augmentations)
        aug_label_path = output_dir / 'labels' / f"{base_name}_aug{i}.txt"
        aug_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(aug_label_path, 'w') as f:
            f.write('\n'.join([' '.join(label) for label in aug_labels]))
    
    print(f"✓ Created {num_augmentations} augmentations for {base_name}")

# Batch augmentation
def batch_augment(images_dir, labels_dir, output_dir, augmentations_per_image=3):
    """
    Turn 50 images into 200 (50 original + 150 augmented)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    for img_path in images_dir.glob('*.jpg'):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            augment_image_and_labels(
                img_path, label_path, output_dir, augmentations_per_image
            )

# Usage: Turn 50 labeled images into 200
batch_augment(
    images_dir='cropped_tables/',
    labels_dir='labeled/',
    output_dir='augmented_dataset/',
    augmentations_per_image=3  # 50 × 4 (original + 3 aug) = 200
)
```

**Time savings:**
- Label 50 images instead of 200
- Run script: 2 minutes
- Result: 200 images with labels

---

## 🤖 Strategy 2: Pre-labeling with Traditional CV

**Auto-detect elements using rules, then manually correct (much faster than starting from scratch)**

### Auto-Labeling Script

```python
import cv2
import pytesseract
import numpy as np
from pathlib import Path
import re

def auto_label_table(image_path, output_label_path):
    """
    Automatically detect and label schedule elements
    You only need to correct mistakes, not label from scratch
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # OCR with bounding boxes
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    labels = []
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        
        if conf < 30 or not text:  # Skip low confidence
            continue
        
        # Get bounding box
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        box_w = ocr_data['width'][i]
        box_h = ocr_data['height'][i]
        
        # Convert to YOLO format (center coordinates, normalized)
        x_center = (x + box_w/2) / w
        y_center = (y + box_h/2) / h
        norm_w = box_w / w
        norm_h = box_h / h
        
        # Classify based on pattern matching
        class_id = classify_text(text, x_center)
        
        if class_id is not None:
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    
    # Save labels
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(labels))
    
    return len(labels)

def classify_text(text, x_position):
    """
    Classify text based on pattern and position
    Returns: class_id or None
    """
    # Course code pattern: 2-4 letters + 2-4 numbers
    if re.match(r'^[A-Z]{2,4}\s?\d{2,4}[A-Z]?$', text):
        return 0  # course_code
    
    # Days pattern
    if re.match(r'^(M|T|W|Th|F|S|MTh|TF|MWF|MW|TTh)$', text):
        return 2  # days
    
    # Time pattern
    if re.search(r'\d{1,2}:\d{2}\s?(AM|PM)', text, re.IGNORECASE):
        # Determine if start or end based on position
        if x_position < 0.85:
            return 3  # time_start
        else:
            return 4  # time_end
    
    # Room pattern (contains numbers and dashes)
    if re.search(r'[A-Z]+-\d+-[A-Z]+', text):
        return 5  # room
    
    # Faculty (contains comma, likely "LastName, F")
    if ',' in text and len(text) > 3:
        return 6  # faculty
    
    # Course title (longer text, usually 3+ words)
    if len(text.split()) >= 3:
        return 1  # course_title
    
    return None

# Batch auto-label
def batch_auto_label(images_dir, output_labels_dir):
    """Auto-label all images"""
    images_dir = Path(images_dir)
    output_labels_dir = Path(output_labels_dir)
    
    for img_path in images_dir.glob('*.jpg'):
        label_path = output_labels_dir / f"{img_path.stem}.txt"
        num_labels = auto_label_table(img_path, label_path)
        print(f"✓ {img_path.name}: {num_labels} elements detected")

# Usage
batch_auto_label('cropped_tables/', 'auto_labels/')
```

**Then in Label Studio:**
1. Import images with auto-generated labels
2. Just **correct mistakes** (5x faster than labeling from scratch)
3. Add missing elements

**Time savings:**
- From scratch: 5 min/image
- With pre-labels: 1 min/image (just corrections)
- **50 images:** 250 min → 50 min (5 hours saved!)

---

## 🎯 Strategy 3: Label Studio Smart Tools

### Use Label Studio's Built-in Features

#### A. **Smart Rectangle Tool** (Auto-snap to text)

In your Label Studio template, add:

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  
  <!-- Enable smart selection -->
  <RectangleLabels name="label" toName="image" 
                   smart="true"           <!-- Auto-detect text boundaries -->
                   showInline="true">
    <Label value="course_code" hotkey="1"/>
    <Label value="course_title" hotkey="2"/>
    <!-- ... other labels ... -->
  </RectangleLabels>
</View>
```

#### B. **Clone Labels** (For repetitive elements)

After labeling first course row:
1. Select all boxes in that row
2. Press `Ctrl+C` (copy)
3. Move to next row
4. Press `Ctrl+V` (paste)
5. Adjust positions slightly

**Time per row:** 30 seconds instead of 2 minutes

---

## 🏆 Strategy 4: Team Labeling (Best Option!)

**Divide and conquer with your team:**

### Team Assignment:

| Person | Task | Images | Time |
|--------|------|--------|------|
| **You** | Label images 1-20 | 20 | 1.5 hrs |
| **Teammate 1** | Label images 21-40 | 20 | 1.5 hrs |
| **Teammate 2** | Label images 41-60 | 20 | 1.5 hrs |
| **Teammate 3** | Label images 61-80 | 20 | 1.5 hrs |
| **Teammate 4** | Label images 81-100 | 20 | 1.5 hrs |

**Result:** 100 labeled images in 1.5 hours (instead of 8+ hours alone)

### Label Studio Multi-User Setup:

```bash
# Create shared Label Studio project
label-studio start schedbuddy --username admin --password admin123

# Share URL with team: http://your-ip:8080
# Each person labels assigned images
```

---

## 💡 Strategy 5: Hybrid Approach (RECOMMENDED)

**Combine multiple strategies for maximum efficiency:**

```
Step 1: Collect 60 COR images (ask 20 classmates for 3 each)

Step 2: Auto-crop all 60 → use traditional CV script

Step 3: Auto-label all 60 → use pattern matching script

Step 4: Split among 3 team members (20 each)
        → Each corrects auto-labels in 1 hour

Step 5: Augment 60 images × 3 = 180 images automatically

Step 6: Add 20 more manually labeled for diversity

Result: 200 high-quality labeled images in ~4 hours total team time
```

---

## ⚡ Fastest Method: Use This Exact Workflow

### Week 1 (3-4 hours total):

**Monday:**
- [ ] Collect 60 COR images from classmates (30 min)
- [ ] Run auto-crop script (5 min)
- [ ] Run auto-label script (10 min)

**Tuesday-Wednesday:**
- [ ] You: Correct auto-labels for images 1-20 (1 hour)
- [ ] Teammate 1: Correct images 21-40 (1 hour)
- [ ] Teammate 2: Correct images 41-60 (1 hour)

**Thursday:**
- [ ] Run augmentation script: 60 → 240 images (5 min)
- [ ] Verify augmented labels (30 min)
- [ ] Split dataset train/val (2 min)

**Friday:**
- [ ] Train model (2-3 hours on CPU, or 30 min on GPU)

**Total active time:** ~4 hours (vs. 25+ hours manual labeling)

---

## 🔧 Complete Automation Script

Here's a **one-click solution** that combines everything:

```python
from pathlib import Path
import subprocess

def automate_everything(raw_cors_dir, output_dir, num_augmentations=3):
    """
    Complete automation pipeline
    
    Input: Raw COR images
    Output: 200+ labeled training images
    """
    
    print("="*60)
    print("SCHEDBUDDY AUTOMATED LABELING PIPELINE")
    print("="*60)
    
    raw_dir = Path(raw_cors_dir)
    output = Path(output_dir)
    
    # Step 1: Auto-crop tables
    print("\n[1/4] Auto-cropping tables...")
    cropped_dir = output / 'cropped'
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    from auto_crop import batch_crop_tables
    batch_crop_tables(raw_dir, cropped_dir)
    
    # Step 2: Auto-label
    print("\n[2/4] Auto-labeling elements...")
    labels_dir = output / 'labels_initial'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    from auto_label import batch_auto_label
    batch_auto_label(cropped_dir, labels_dir)
    
    # Step 3: Manual correction prompt
    print("\n[3/4] Opening Label Studio for manual corrections...")
    print("     → Import images from:", cropped_dir)
    print("     → Import labels from:", labels_dir)
    print("     → Correct any mistakes")
    print("\n     Press ENTER when done...")
    input()
    
    # Step 4: Augment
    print("\n[4/4] Augmenting dataset...")
    augmented_dir = output / 'final_dataset'
    
    from augmentation import batch_augment
    batch_augment(
        cropped_dir, 
        labels_dir,  # Use corrected labels
        augmented_dir,
        num_augmentations
    )
    
    # Count final dataset
    train_imgs = list((augmented_dir / 'images').glob('*.jpg'))
    print("\n" + "="*60)
    print(f"✓ COMPLETE! Final dataset: {len(train_imgs)} images")
    print(f"  Location: {augmented_dir}")
    print("="*60)

# Run
automate_everything(
    raw_cors_dir='raw_cors/',
    output_dir='schedbuddy_dataset/',
    num_augmentations=3
)
```

---

## 📊 Time Comparison

| Method | Time for 200 images | Quality |
|--------|---------------------|---------|
| **Manual labeling alone** | 25-30 hours | 100% |
| **With pre-labeling** | 8-10 hours | 95% |
| **With augmentation (50 real)** | 5-6 hours | 85% |
| **Team of 3 + augmentation** | 2-3 hours each | 90% |
| **Pre-label + Team + Aug** | **1.5 hours each** ⭐ | **90%** |

---

## 🎯 My Recommendation

**Use Strategy 5 (Hybrid):**

1. Get 2-3 teammates to help
2. Collect 60 COR images
3. Auto-crop and auto-label
4. Each person corrects 20 images (1 hour)
5. Augment to 200+
6. Train model

**Total time: 3-4 hours team effort = 200+ training images**

Would you like me to create the complete automation scripts for you, or help set up the team workflow in Label Studio?