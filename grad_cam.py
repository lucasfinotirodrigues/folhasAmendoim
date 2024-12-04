from ultralytics import YOLO
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

def apply_grad_cam(image_path, model_path, output_path="grad_cam_result.jpg"):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Normalizar a imagem
    input_image = cv2.resize(image, (640, 640)) / 255.0
    input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float()

    # Carregar o modelo em modo de inferência
    model = YOLO(model_path)  # YOLO carregado com o peso treinado
    model.fuse()  # Otimizar o modelo para inferência

    # Configurar o Grad-CAM
    target_layer = model.model.model[-2]  # Penúltima camada (ajuste se necessário)
    cam = GradCAM(model=model.model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    # Aplicar o Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
    grayscale_cam = grayscale_cam[0, :]

    # Sobrepor Grad-CAM na imagem original
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    # Salvar o resultado
    cv2.imwrite(output_path, np.uint8(visualization * 255))
    print(f"Grad-CAM salvo em: {output_path}")

# Chamada da função
apply_grad_cam(
    image_path="dataset/images/teste/mancha-preta-precoce/mancha-precoce1.jpg",
    model_path="dataset/runs/detect/train5/weights/best.pt",
    output_path="grad_cam_result.jpg"
)
