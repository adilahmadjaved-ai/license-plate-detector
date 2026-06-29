from ultralytics import YOLO
import easyocr
import cv2
import re

# --------------------------
# Load Models
# --------------------------
model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

# --------------------------
# Load Image
# --------------------------
image_path = "car.jpg"
image = cv2.imread(image_path)

# --------------------------
# Detect License Plate
# --------------------------
results = model(image)

# --------------------------
# Process All Detections
# --------------------------
for box in results[0].boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Crop plate
    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        continue

    # --------------------------
    # Preprocessing
    # --------------------------
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # --------------------------
    # OCR
    # --------------------------
    ocr_result = reader.readtext(
        gray,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    plate_text = "UNKNOWN"

    if len(ocr_result) > 0:
        plate_text = ocr_result[0][1]
        plate_text = re.sub(
            r'[^A-Z0-9]',
            '',
            plate_text.upper()
        )

    # --------------------------
    # Draw Results
    # --------------------------
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{plate_text} ({conf:.2f})"

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# --------------------------
# Save Output
# --------------------------
cv2.imwrite("output.jpg", image)

print("Done!")
print("Saved as output.jpg")