import cv2
import easyocr
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO

# Hardcode your trained YOLO model path here
MODEL_PATH = "best.pt"  # <<--- change this

def detect_license_plate(image_path, padding=10, show_image=True, use_gpu=True):
    # Load YOLO model once
    model = YOLO(MODEL_PATH)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Run YOLO inference
    results = model.predict(img)
    boxes = results[0].boxes

    if not boxes:
        print("No license plate detected.")
        return

    # Extract bounding box
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Add padding and clamp within image bounds
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    cropped = img[y1:y2, x1:x2]

    # Show cropped image
    if show_image:
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title("Cropped License Plate")
        plt.axis("off")
        plt.show()

    # OCR
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    ocr_result = reader.readtext(cropped)

    if ocr_result:
        plate_text = ocr_result[0][-2]
        print("✅ Detected Plate Number:", plate_text)
    else:
        print("❌ OCR failed to detect text.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Detection using YOLO and EasyOCR")
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--padding', type=int, default=10, help='Padding around the detected plate')
    parser.add_argument('--no-show', action='store_true', help="Don't show cropped plate image")
    parser.add_argument('--cpu', action='store_true', help='Force OCR to run on CPU')

    args = parser.parse_args()

    detect_license_plate(
        image_path=args.image,
        padding=args.padding,
        show_image=not args.no_show,
        use_gpu=not args.cpu
    )
