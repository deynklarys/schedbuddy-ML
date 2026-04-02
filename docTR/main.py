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

# %% Manual Visualization (Fixed)
import matplotlib.pyplot as plt
from doctr.utils.reconstitution import synthesize_page

# 1. Correct the imports (They live in different places now)
from doctr.utils.visualization import visualize_page

# 2. Iterate through the results manually
for page in result.pages:
    # A. The Overlay (Boxes on Image)
    # We pass the exported data and the actual image array
    visualize_page(page.export(), page.page, interactive=True)
    plt.title("OCR Overlay")
    plt.show()

    # B. The Reconstitution (Text on White Page)
    # We pass the exported data to rebuild the page from scratch
    synth_img = synthesize_page(page.export(), draw_proba=True)
    plt.imshow(synth_img)
    plt.axis("off")
    plt.title("Reconstructed Text")
    plt.show()
