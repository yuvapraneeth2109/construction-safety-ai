from ultralytics import YOLO

model = YOLO("models/edgee.pt")
print(model.names)
