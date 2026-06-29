import cv2

def draw(image, box, text, conf):
    x1, y1, x2, y2 = box

    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(
        image,
        f"{text} ({conf:.2f})",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )