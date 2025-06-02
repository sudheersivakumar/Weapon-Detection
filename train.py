from ultralytics import YOLO

# Load a YOLOv8 model (use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', or 'yolov8x.pt' based on your requirement)
model = YOLO("yolov8n.pt")  # You can change this to 'yolov8s.pt' or other versions

# Train the model
if __name__ == '__main__':
    model.train(
    data="C:/Users/sudhe/OneDrive/Documents/wep1/data.yaml",  # Path to dataset config
    epochs=200,       # Number of training epochs
    batch=4,         # Batch size (adjust based on GPU)
    imgsz=640,        # Image size for training
    workers=4,        # CPU workers
    lr0=0.001,        # Initial learning rate
    cos_lr=True,      # Use cosine learning rate decay
    optimizer="SGD",  # Optimizer (SGD, AdamW, etc.)
    weight_decay=0.0005,  # Regularization
    augment=True,     # Data augmentation
    mosaic=1.0,       # Mosaic augmentation
    mixup=0.2,        # MixUp augmentation
    label_smoothing=0.1,  # Prevent overconfidence
    patience=50       # Early stopping patience  # Using GPU
    )

