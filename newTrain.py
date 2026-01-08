from ultralytics import YOLO

# Load a pretrained model (YOLO11 or the newer YOLO26)
model = YOLO("yolo11n.pt") 
if __name__ == '__main__':
# Train the model
    results = model.train(
        data="./dataset/data.yaml", 
        epochs=100, 
        imgsz=640, 
        device=0  # use 'device=[0, 1]' for multi-GPU
    )