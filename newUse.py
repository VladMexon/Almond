from ultralytics import YOLO

# 1. Load your custom trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# 2. Run inference on a new source (image, folder, video, or webcam)
results = model.predict(source="Phellinus-igniarius-2015-07-07-IMG_0659-220.jpg", save=True, conf=0.5)

# 3. Process the results
for result in results:
    boxes = result.boxes  # Bounding box outputs
    result.show()         # Display the image with detections
