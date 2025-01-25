import torch
import numpy as np
import cv2
from ultralytics import YOLO

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registra os hooks para capturar gradientes e ativações
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx):
        # Forward pass
        outputs = self.model(input_image)

        # Calcula os gradientes para a classe alvo
        self.model.zero_grad()
        target = outputs[0, class_idx]
        target.backward(retain_graph=True)

        # Calcula o Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1).squeeze()

        # Normaliza para a escala [0, 1]
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().detach().numpy()

# Carregue o modelo YOLO
model = YOLO('yolov8n.pt')
target_layer = dict(model.model.named_modules()).get('model.22.cv3.2.0')
grad_cam = GradCAM(model.model, target_layer=target_layer)


# Imagem de exemplo (formato tensor)
input_image = torch.randn((1, 3, 640, 640))  # Substitua por uma imagem real preprocessada
cam = grad_cam.generate_cam(input_image, class_idx=0)

# Exibir o mapa de calor
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(input_image.numpy()[0].transpose(1, 2, 0), 0.5, heatmap, 0.5, 0)
cv2.imshow('Grad-CAM', overlay)
cv2.waitKey(0)



# from ultralytics import YOLO

# # Carregar o modelo YOLO
# model = YOLO('yolov8n.pt')

# # Listar todas as camadas do modelo
# for name, module in model.model.named_modules():
#     print(name)

#     target_layer = dict(model.model.named_modules()).get('model.22.cv3.2.0')
# if target_layer is None:
#     print("Camada não encontrada.")
# else:
#     print("Camada encontrada:", target_layer)

