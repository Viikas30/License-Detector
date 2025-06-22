import cv2
import numpy as np
import imutils
import easyocr
import sys
from matplotlib import pyplot as plt

def detect_license_plate(image_path, show_steps=False, save_annotated=False, output_path="annotated_image.jpg"):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Look for a contour with 4 sides (likely to be the plate)
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return "License plate not found"

    # Mask and crop the license plate
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # OCR
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped_image)

    if len(result) == 0:
        return "Text not detected"

    # Get detected text
    text = result[0][-2]

    # Annotate image with text and bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org=(location[0][0][0], location[1][0][1] + 60),
                fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

    # Save annotated image
    if save_annotated:
        cv2.imwrite(output_path, img)

    # Show images if needed
    if show_steps:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Edged")
        plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("Cropped Plate")
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 3)
        plt.title("Annotated Result")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    return text

# --- Command-line usage ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python license_plate_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detected_plate = detect_license_plate(image_path, show_steps=True, save_annotated=True)
    print("Detected Plate Number:", detected_plate)
