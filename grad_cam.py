from ultralytics import YOLO
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np

def apply_grad_cam(image_path, model_path, output_path="grad_cam_result.jpg"):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Preprocessar a imagem
    input_image = cv2.resize(image, (640, 640)) / 255.0
    input_image = np.float32(input_image)
    input_tensor = preprocess_image(input_image, mean=[0, 0, 0], std=[1, 1, 1])

    # Configurar o dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar o modelo
    model = YOLO(model_path)
    model.fuse()  # Otimizar o modelo para inferência
    model.model.to(device).eval()  # Definir o modelo como inferência

    # Envolver o modelo para compatibilidade com Grad-CAM
    class CustomYOLOModelWrapper(torch.nn.Module):
        def __init__(self, yolov5_model):
            super(CustomYOLOModelWrapper, self).__init__()
            self.model = yolov5_model

        def forward(self, x):
            return self.model(x)[0]  # Retorna apenas previsões

    wrapped_model = CustomYOLOModelWrapper(model.model)

    # Configurar o Grad-CAM
    target_layer = model.model.model[-2]  # Penúltima camada
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])

    # Certifique-se de que o tensor requer gradiente
    input_tensor = input_tensor.to(device).requires_grad_()

    # Passar a imagem através do modelo e obter as previsões
    with torch.no_grad():
        predictions = model(input_tensor)  # Saída: caixa de predição, classe, etc.
    
    # Verifique o conteúdo de predictions
    print("Predictions:", predictions)
    
    # Acessar as caixas e as probabilidades
    boxes = predictions[0].boxes  # Caixa de predição
    probs = predictions[0].probs  # Probabilidades de cada classe

    print("Boxes:", boxes)
    print("Probabilities:", probs)

    # Se não houver caixas detectadas, não podemos aplicar o Grad-CAM
    if boxes.shape[0] == 0:
        print("Nenhuma caixa detectada.")
        return

    # A classe e confiança estão nos índices 5 e acima (depende do número de classes)
    # Ajuste conforme a estrutura do YOLO
    class_confidences = probs  # Confiança de cada classe
    predicted_class = torch.argmax(class_confidences, dim=1)  # Classe com maior probabilidade

    # Obter o índice da classe
    target_category = predicted_class.item()  # Converter para valor escalar

    # Aplicar Grad-CAM para o target escolhido
    grayscale_cam = cam(input_tensor=input_tensor, targets=[target_category])
    grayscale_cam = grayscale_cam[0, :]  # Pegar a primeira previsão de Grad-CAM

    # Visualizar e salvar o Grad-CAM
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, np.uint8(visualization * 255))
    print(f"Grad-CAM salvo em: {output_path}")

# Chamada da função
apply_grad_cam(
    image_path="dataset/images/teste/mancha-preta-precoce/mancha-precoce1.jpg",
    model_path="runs/detect/train4/weights/best.pt",
    output_path="grad_cam_result.jpg"
)
