import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Função para carregar o modelo YOLO
def load_yolo_model(model_path):
    model = YOLO(model_path)  # Carrega o modelo YOLOv5
    return model

# Função para realizar a inferência e obter as detecções
def run_inference(model, img_path):
    results = model(img_path)  # Faz a predição
    return results

# Função para gerar Grad-CAM
def make_gradcam_heatmap(yolo_model, img_path):
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))  # Redimensiona para o formato esperado pelo YOLO
    img = img / 255.0  # Normaliza os pixels
    img = np.expand_dims(img, axis=0)  # Adiciona dimensão de batch

    # Obtém a última camada convolucional do modelo YOLOv5
    last_conv_layer = yolo_model.model.model[-2]  # Última camada convolucional

    # Calcula o Grad-CAM
    with torch.no_grad():
        output = yolo_model(img)  # Faz a predição
    gradients = torch.autograd.grad(outputs=output, inputs=last_conv_layer, grad_outputs=torch.ones_like(output))

    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])
    last_conv_layer_output = last_conv_layer(img)
    last_conv_layer_output = last_conv_layer_output.squeeze().detach().cpu().numpy()

    heatmap = np.mean(last_conv_layer_output, axis=0)
    heatmap = np.maximum(heatmap, 0)  # Aplica ReLU
    heatmap /= np.max(heatmap)  # Normaliza entre 0 e 1

    return heatmap

# Função para sobrepor Grad-CAM à imagem original
def overlay_heatmap(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Redimensiona para o tamanho da imagem original
    heatmap = np.uint8(255 * heatmap)  # Converte para escala de 0 a 255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplica um mapa de cores

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Sobrepõe a imagem original com o heatmap
    cv2.imwrite(output_path, superimposed_img)  # Salva a imagem resultante

# Função principal
def main():
    model_path = "yolov5su.pt"  # Modelo YOLO treinado
    image_dir = "images"  # Diretório das imagens
    output_dir = "gradcam_results"
    os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se não existir

    # Carrega o modelo YOLO
    yolo_model = load_yolo_model(model_path)

    # Obtém a lista de imagens
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    print(f"Número de imagens encontradas: {len(image_paths)}")
    if len(image_paths) == 0:
        print("Nenhuma imagem encontrada.")
        return

    for img_path in image_paths:
        print(f"Processando {img_path}...")

        # Executa a inferência no YOLO
        results = run_inference(yolo_model, img_path)

        # Gera o heatmap do Grad-CAM
        heatmap = make_gradcam_heatmap(yolo_model, img_path)

        # Define o nome da imagem de saída
        output_image_path = os.path.join(output_dir, f"gradcam_{os.path.basename(img_path)}")

        # Sobrepõe e salva a imagem com o heatmap
        overlay_heatmap(img_path, heatmap, output_image_path)

        print(f"Grad-CAM salvo em: {output_image_path}")

if __name__ == "__main__":
    main()
