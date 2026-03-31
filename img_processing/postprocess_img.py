# %% Setup
import cv2
import numpy as np
from matplotlib import pyplot as plt

# image_file = "./raw_samples/5ef068b5-113_table_1(input).jpg"
# image_file = "./raw_samples/87ef5a9f-25_table_1(input1).jpg"
# image_file = "./raw_samples/32c2ea01-196_table_1(input2).jpg"
image_file = "./raw_samples/8c5da3c8-105_table_1(input3).jpg"
img = cv2.imread(image_file)
print(f"Setup complete. Image file = {image_file}")


# %% Display with actual size
# https://stackoverflow.com/questions/28816046/


def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis("off")

    # Display the image.
    ax.imshow(im_data, cmap="gray")

    plt.show()


display(image_file)


# %% Inversion
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("temp/inverted.jpg", inverted_image)

display("temp/inverted.jpg")


# %% Binarization
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


greyed = grayscale(img)
cv2.imwrite("temp/greyed.jpg", greyed)

display("temp/greyed.jpg")

# %%
thresh, im_bw = cv2.threshold(greyed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("temp/bw_image.jpg", im_bw)

display("temp/bw_image.jpg")


# %% Noise Removal
def rm_noise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones(
        (
            1,
            1,
        ),
        np.uint8,
    )
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


quiet = rm_noise(im_bw)
cv2.imwrite("temp/quiet.jpg", quiet)

display("temp/quiet.jpg")


# %% Erosion
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


eroded_image = thin_font(quiet)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)

display("temp/eroded_image.jpg")


# %% Dilation
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


dilated_image = thick_font(quiet)
cv2.imwrite("temp/dilated_image.jpg", dilated_image)

display("temp/dilated_image.jpg")

# %% Automatic deskewing
# https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
skewed = cv2.imread(image_file)
display(image_file)


# %% Use only on skewed images
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


fixed = deskew(skewed)
cv2.imwrite("temp/deskewed.jpg", fixed)

display("temp/deskewed.jpg")
