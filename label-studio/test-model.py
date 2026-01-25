# yolo detect predict model=runs/detect/train/weights/best.pt source=datasets/validation/images save=True

import glob
from IPython.display import Image, display
for image_path in glob.glob(f'.\\label-studio\\runs\\detect\\predict\\*.jpg')[:10]:
  display(Image(filename=image_path, height=400))
  print('\n')
