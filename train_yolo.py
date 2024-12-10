from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'/home/ciag/projetosPessoais/folhasAmendoim/dataset.yaml', 
        epochs=5,            
        imgsz=640,            
        batch=16,             
        device='cpu'              
    )

if __name__ == '__main__':
    train_yolo()
