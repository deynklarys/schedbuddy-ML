# %% Imports
import json

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

print("Imports ready.")
# %% Image paths
img1 = "./samples/input00.jpg"
img2 = "./samples/input01.jpg"
img3 = "./samples/input3.jpg"
img4 = "./samples/input4.jpg"

print("Image paths ready.")

# %%
model = ocr_predictor(pretrained=True)  # default pretrained model
imageFile = DocumentFile.from_images(img1)

print("Setup complete.")

# %%
result = model(imageFile)
result_json = result.export()
print(json.dumps(result_json, indent=4))

result.show(imageFile)
