import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Estadísticas estándar de ImageNet.
# Al usar modelos preentrenados en ImageNet, debemos normalizar nuestros tensores
# con exactamente la misma media y desviación estándar para que los filtros
# convolucionales iniciales reciban distribuciones de píxeles conocidas.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """
    Define los pipelines de preprocesamiento y Data Augmentation.
    
    ¿Por qué separar train de val/test?
    En entrenamiento aplicamos perturbaciones geométricas y fotométricas aleatorias
    para evitar el sobreajuste (overfitting) y lograr invarianza a rotación, escala e iluminación.
    En validación y prueba, la evaluación debe ser determinista (sin aleatoriedad).
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Transformaciones geométricas: una hoja puede fotografiarse en cualquier ángulo
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Transformaciones fotométricas: simula cambios de luz solar o sombra en el campo
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # Convierte imagen PIL (H, W, C) en rango [0, 255] a Tensor (C, H, W) en rango [0.0, 1.0]
        transforms.ToTensor(),
        # Normalización Z-score por canal RGB
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # RandomErasing simula oclusiones (ej. dedos, insectos o gotas de agua tapando la hoja)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transforms, eval_transforms


class TransformedSubset(torch.utils.data.Dataset):
    """
    Wrapper necesario en PyTorch cuando dividimos un ImageFolder con random_split
    o índices estratificados, permitiendo aplicar transforms distintos a train y val/test
    sin que se sobrescriban entre sí.
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def create_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 4):
    """
    Crea los DataLoaders estratificados de Train (70%), Validation (15%) y Test (15%),
    además de calcular los pesos de clase para combatir el desbalanceo.
    """
    base_dataset = datasets.ImageFolder(root=data_dir)
    targets = np.array(base_dataset.targets)
    indices = np.arange(len(targets))
    class_names = base_dataset.classes

    # Split estratificado: garantiza que todas las enfermedades tengan la misma proporción
    # en entrenamiento, validación y prueba.
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, targets, test_size=0.30, stratify=targets, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y_temp, random_state=42
    )

    train_tf, eval_tf = get_transforms(img_size=img_size)

    train_dataset = TransformedSubset(Subset(base_dataset, train_idx), transform=train_tf)
    val_dataset = TransformedSubset(Subset(base_dataset, val_idx), transform=eval_tf)
    test_dataset = TransformedSubset(Subset(base_dataset, test_idx), transform=eval_tf)

    # Cálculo de pesos inversamente proporcionales a la frecuencia de cada clase:
    # W_c = N / (C * N_c). Penaliza más los errores en enfermedades raras.
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_names)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # pin_memory=True acelera la transferencia de tensores de RAM (CPU) a VRAM (GPU)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader, class_names, class_weights_tensor