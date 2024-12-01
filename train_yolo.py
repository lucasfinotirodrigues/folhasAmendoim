from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'c:\Users\lucas\OneDrive\Área de Trabalho\folhasAmendoim\dataset.yaml', 
        epochs=50,            
        imgsz=640,            
        batch=16,             
        device='cpu'              
    )

if __name__ == '__main__':
    train_yolo()
