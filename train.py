from ultralytics import YOLO


if __name__ == '__main__':
    # Зареждаме малък модел
    model = YOLO('yolov8n.pt')

    # Стартираме обучението
    model.train(data='data.yaml', epochs=25, imgsz=640, device='cpu')
