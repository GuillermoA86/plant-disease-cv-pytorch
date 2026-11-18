import json
import torch
import onnx
import onnxruntime as ort
import numpy as np

from src.model import PlantDiseaseClassifier


def export_to_onnx():
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    checkpoint = torch.load("models/best_plant_model.pth", map_location="cpu")
    model = PlantDiseaseClassifier(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "models/plant_disease_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["logits"],
        dynamic_axes={"input_image": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )

    # Verificación de integridad del modelo exportado
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Comparar salida de PyTorch vs ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)[0]

    with torch.no_grad():
        torch_outs = model(dummy_input).numpy()

    np.testing.assert_allclose(torch_outs, ort_outs, rtol=1e-03, atol=1e-05)
    print(f"Modelo exportado y verificado exitosamente en: {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()