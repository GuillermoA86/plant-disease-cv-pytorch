import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.model import PlantDiseaseClassifier


def get_device():
    """Selecciona automáticamente el acelerador de hardware disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Aceleración GPU en Apple Silicon (M1/M2/M3)
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    """Ejecuta una época completa de entrenamiento."""
    model.train()  # Activa Dropout y actualización de estadísticas en BatchNorm
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # 1. Reiniciar gradientes acumulados de la iteración anterior
        # set_to_none=True es más rápido y consume menos memoria que poner ceros
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward Pass con Automatic Mixed Precision (FP16/FP32)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 3. Backward Pass (cálculo de gradientes mediante backpropagation)
        scaler.scale(loss).backward()

        # 4. Gradient Clipping: evita la explosión de gradientes durante el fine-tuning
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Actualización de pesos
        scaler.step(optimizer)
        scaler.update()

        # Métricas
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / total, correct / total


@torch.no_grad()  # Desactiva el motor de autograd: ahorra memoria y acelera la inferencia
def validate(model, dataloader, criterion, device):
    """Evalúa el modelo en el conjunto de validación o prueba."""
    model.eval()  # Congela BatchNorm y desactiva Dropout
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    DATA_DIR = "data/PlantVillage"
    SAVE_DIR = "models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS_HEAD = 2       # Fase 1: Entrenar solo la cabeza clasificadora
    EPOCHS_FINETUNE = 3   # Fase 2: Descongelar bloques superiores y afinar pesos
    LR_HEAD = 1e-3
    LR_FINETUNE = 1e-4

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Entrenando en dispositivo: {device} | AMP habilitado: {use_amp}")

    train_loader, val_loader, _, class_names, class_weights = create_dataloaders(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=0
    )

    # Guardamos el mapeo de índices a nombres de clases para la inferencia y Streamlit
    with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    model = PlantDiseaseClassifier(num_classes=len(class_names), freeze_backbone=True).to(device)

    # Loss ponderada por frecuencia de clase + Label Smoothing (0.1) para evitar
    # que la red se vuelva excesivamente confiada en sus predicciones y mejore generalización.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # --- FASE 1: Feature Extraction (Cabeza nueva, backbone congelado) ---
    print("\n--- FASE 1: Entrenando cabeza de clasificación ---")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_HEAD)

    for epoch in range(EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Época {epoch+1}/{EPOCHS_HEAD} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # --- FASE 2: Fine-Tuning (Descongelamiento parcial con Learning Rate bajo) ---
    print("\n--- FASE 2: Fine-Tuning de bloques convolucionales profundos ---")
    model.unfreeze_backbone(unfreeze_from_block=5)
    
    # Discriminative Learning Rates: LR menor para el backbone preentrenado,
    # LR mayor para la cabeza clasificadora.
    optimizer = optim.AdamW([
        {"params": model.backbone.features.parameters(), "lr": LR_FINETUNE * 0.2},
        {"params": model.backbone.classifier.parameters(), "lr": LR_FINETUNE}
    ], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

    patience = 3
    epochs_no_improve = 0

    for epoch in range(EPOCHS_FINETUNE):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Época {epoch+1}/{EPOCHS_FINETUNE} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping activado tras {patience} épocas sin mejora.")
                break

    # Guardamos el mejor checkpoint
    model.load_state_dict(best_model_wts)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(class_names),
        "best_val_acc": best_val_acc
    }, os.path.join(SAVE_DIR, "best_plant_model.pth"))
    print(f"\nEntrenamiento finalizado. Mejor Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()