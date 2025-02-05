import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO("runs/detect/train4/weights/best.pt")

def find_last_conv_layer(model):
    layers = list(model.model.named_children())  # Converte o gerador para lista
    for name, module in reversed(layers):  # Agora pode usar reversed()
        print(name, "->", module)
        if isinstance(module, torch.nn.Conv2d):  
            return name
    return None

# Obter a última camada convolucional do modelo
last_conv_layer = find_last_conv_layer(model)

if last_conv_layer is None:
    raise ValueError("Nenhuma camada convolucional encontrada no modelo!")

# Função para aplicar Grad-CAM
def grad_cam_yolo(model, img_path, target_layer_name):
    # Carregar a imagem
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Executar inferência no YOLO
    results = model(img_path)  
    result = results[0]  # Pega o primeiro resultado

    # Obtém a camada alvo
    layer = dict(model.model.named_children())[target_layer_name]
    
    # Função para capturar gradientes e ativações
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Adicionar hooks na camada convolucional
    layer.register_full_backward_hook(backward_hook)
    layer.register_forward_hook(forward_hook)

    # Verifica se houve alguma detecção
    if not result.boxes:
        print(f"Nenhuma detecção na imagem {img_path}")
        return img_rgb, None, result

    # Obtém a classe com maior confiança
    preds = result.probs
    class_idx = preds.argmax().item()
    loss = preds[class_idx]

    # Calcula os gradientes via backpropagation
    model.zero_grad()
    loss.backward()

    # Obtém os mapas de ativação e gradientes
    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    # Calcula a média dos gradientes por canal
    weights = np.mean(grad, axis=(1, 2))

    # Combina os mapas de ativação com os gradientes
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # Normaliza os valores entre 0 e 1
    cam = np.maximum(cam, 0)
    cam /= cam.max()

    # Redimensiona para o tamanho da imagem original
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    return img_rgb, cam, result

# Função para sobrepor o Grad-CAM na imagem original
def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

# Loop para processar as imagens
for i in range(1, 6):
    img_path = f"mancha-tardia{i}.jpg"
    img_rgb, heatmap, result = grad_cam_yolo(model, img_path, last_conv_layer)

    if heatmap is not None:
        # Aplicar o Grad-CAM nas áreas detectadas pelo YOLO
        overlayed_img = overlay_heatmap(img_rgb, heatmap)

        # Desenhar bounding boxes
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlayed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostrar a imagem
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"Imagem Original {i}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(overlayed_img)
        plt.title(f"Grad-CAM {i}")
        plt.axis("off")

        plt.show()
