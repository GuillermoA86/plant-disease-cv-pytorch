import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import create_dataloaders
from src.model import PlantDiseaseClassifier
from src.train import get_device


class GradCAM:
    """
    Implementación nativa de Grad-CAM (Gradient-weighted Class Activation Mapping)
    usando PyTorch Hooks.
    
    ¿Por qué es clave en una entrevista?
    Demuestra que no tratas la red como una caja negra: extraes los gradientes que fluyen
    hacia la última capa convolucional para resaltar qué píxeles exactos de la hoja
    (ej. manchas necróticas y no el fondo) determinaron el diagnóstico.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registramos hooks en el grafo computacional dinámico de PyTorch
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        # Global Average Pooling sobre los gradientes para obtener la importancia de cada canal
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # Aplicamos ReLU porque solo nos interesan las características que influyen positivamente
        cam = F.relu(cam).cpu().numpy()

        # Normalizamos entre 0 y 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def overlay_heatmap(original_img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpone el mapa de calor Grad-CAM sobre la imagen original."""
    h, w, _ = original_img_np.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def evaluate_test_set():
    device = get_device()
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    _, _, test_loader, _, _ = create_dataloaders("data/PlantVillage", batch_size=32, num_workers=0)

    checkpoint = torch.load("models/best_plant_model.pth", map_location=device)
    model = PlantDiseaseClassifier(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("=== REPORTE DE CLASIFICACIÓN EN TEST ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión - Diagnóstico de Plantas")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    print("Matriz de confusión guardada en models/confusion_matrix.png")


if __name__ == "__main__":
    evaluate_test_set()