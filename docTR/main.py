# %% Imports
import matplotlib

matplotlib.use("TkAgg")  # This tells Python to open a separate window
import json

import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.reconstitution import synthesize_page
from doctr.utils.visualization import visualize_page

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

# %% Manual Visualization
# 1. Turn on Interactive Mode BEFORE starting
plt.ion()

for i, page in enumerate(result.pages):
    # A. The Overlay
    # Creating a unique figure number ensures windows don't overwrite each other
    plt.figure(i * 2)
    visualize_page(page.export(), page.page, interactive=True)
    plt.title(f"Page {i} - OCR Overlay")

    # B. The Reconstitution
    plt.figure(i * 2 + 1)
    synth_img = synthesize_page(page.export(), draw_proba=True)
    plt.imshow(synth_img)
    plt.axis("off")
    plt.title(f"Page {i} - Reconstructed")

# 2. This is the "Breath" that prevents the freeze
# It tells Python to run the GUI event loop for a moment
plt.show(block=True)
