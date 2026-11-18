import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.dataset import get_transforms
from src.model import PlantDiseaseClassifier
from src.evaluate import GradCAM, overlay_heatmap

st.set_page_config(page_title="PhytoVision AI - Diagnóstico Botánico", layout="wide")


@st.cache_resource
def load_model_and_classes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    checkpoint = torch.load("models/best_plant_model.pth", map_location=device)
    model = PlantDiseaseClassifier(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Apuntamos Grad-CAM al último bloque convolucional de EfficientNet-B0
    target_layer = model.backbone.features[-1]
    gradcam = GradCAM(model, target_layer)
    return model, class_names, gradcam, device


st.title("PhytoVision AI: Diagnóstico de Enfermedades en Plantas con PyTorch")
st.markdown(
    "Sistema de Visión por Computadora basado en **EfficientNet-B0** con **Transfer Learning**, "
    "calibración de probabilidades y explicabilidad visual mediante **Grad-CAM**."
)

try:
    model, class_names, gradcam, device = load_model_and_classes()
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado en `models/best_plant_model.pth`. Ejecuta primero `python -m src.train`.")
    st.stop()

uploaded_file = st.file_uploader("Sube una fotografía de la hoja de la planta (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    _, eval_tf = get_transforms(img_size=224)

    # Preprocesamiento y creación de dimensión batch: (3, 224, 224) -> (1, 3, 224, 224)
    input_tensor = eval_tf(image).unsqueeze(0).to(device)

    # Inferencia y generación de Grad-CAM
    # Nota: No usamos torch.no_grad() aquí porque Grad-CAM necesita calcular gradientes
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1).squeeze().detach().cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    heatmap = gradcam.generate_heatmap(input_tensor, class_idx=top_idx)
    overlay_img = overlay_heatmap(np.array(image), heatmap, alpha=0.45)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Atención de la Red (Grad-CAM)")
        st.image(overlay_img, use_container_width=True, caption="Las zonas rojas indican los píxeles determinantes para el diagnóstico.")

    with col3:
        st.subheader("Resultado del Diagnóstico")
        raw_label = class_names[top_idx]
        formatted_label = raw_label.replace("___", " — ").replace("_", " ")
        confidence = probabilities[top_idx] * 100

        if "healthy" in raw_label.lower():
            st.success(f"**Estado:** {formatted_label}")
        else:
            st.warning(f"**Diagnóstico:** {formatted_label}")

        st.metric("Confianza del Modelo", f"{confidence:.2f}%")

        # Top-5 predicciones
        top5_indices = np.argsort(probabilities)[::-1][:5]
        df_probs = pd.DataFrame({
            "Diagnóstico": [class_names[i].replace("___", " — ").replace("_", " ") for i in top5_indices],
            "Probabilidad (%)": [round(float(probabilities[i]) * 100, 2) for i in top5_indices]
        })
        st.bar_chart(df_probs.set_index("Diagnóstico"))