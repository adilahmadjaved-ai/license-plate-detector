from ultralytics import YOLO

model = YOLO("best.pt")

def detect(image):
    return model(image)