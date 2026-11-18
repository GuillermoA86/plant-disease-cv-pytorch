import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseClassifier(nn.Module):
    """
    Arquitectura basada en EfficientNet-B0 con Transfer Learning.
    
    ¿Por qué EfficientNet-B0?
    1. Escalado compuesto: balancea profundidad, ancho y resolución.
    2. Bloques MBConv con atención Squeeze-and-Excitation (SE), ideales para captar
       texturas finas (manchas fúngicas, necrosis, decoloración) en hojas.
    3. Ligero (~5.3M parámetros), rápido para inferencia en tiempo real y fácil de entrenar.
    """
    def __init__(self, num_classes: int, freeze_backbone: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        # Cargamos pesos preentrenados en ImageNet1K_V1
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Congelamos las capas extractoras de características (feature extractor)
        # en la primera etapa para no destruir los filtros de bordes y texturas aprendidos.
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Obtenemos la dimensión de salida del backbone antes del clasificador original (1280)
        in_features = self.backbone.classifier[1].in_features

        # Reemplazamos la cabeza de clasificación original de 1000 clases por una
        # adaptada a nuestro dominio con BatchNorm y Dropout para regularización.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),  # SiLU (Swish) es la activación nativa de EfficientNet
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def unfreeze_backbone(self, unfreeze_from_block: int = 5):
        """
        Descongela los últimos bloques convolucionales para Fine-Tuning fino.
        Las primeras capas detectan bordes genéricos; las últimas detectan patrones
        complejos específicos de las enfermedades de plantas.
        """
        for idx, child in enumerate(self.backbone.features.children()):
            if idx >= unfreeze_from_block:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retorna logits crudos (sin Softmax), ya que CrossEntropyLoss aplica
        # LogSoftmax internamente por estabilidad numérica.
        return self.backbone(x)