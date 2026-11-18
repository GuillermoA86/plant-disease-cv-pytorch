Markdown# PhytoVision AI: Diagnóstico de Enfermedades en Plantas con PyTorch

Sistema completo de **Computer Vision** desarrollado en **PyTorch puro** para clasificar enfermedades foliares en cultivos agrícolas a partir de imágenes RGB. El proyecto abarca desde la ingeniería de datos y el entrenamiento en dos fases (*Transfer Learning* + *Fine-Tuning*) hasta la explicabilidad visual (**Grad-CAM**), exportación a **ONNX** para producción y un tablero interactivo en **Streamlit**.

---

## 1. Dataset Utilizado

Se utiliza el subconjunto de cultivos solanáceos (**tomate, papa y pimiento**) del dataset público **PlantVillage** (`emmarex/plantdisease` en Kaggle), compuesto por **~20,600 imágenes** distribuidas en **15 clases** (plantas sanas y diversas patologías fúngicas, bacterianas y virales).

Este subconjunto representa un caso de uso realista y exigente en agricultura de precisión, ya que incluye patologías visualmente muy similares entre especies emparentadas (como *Early Blight* vs. *Late Blight* en papa y tomate), poniendo a prueba la capacidad discriminativa de los filtros convolucionales.

### Descarga y Organización

1. Descarga el dataset mediante la API de Kaggle o `kagglehub`:
   ```bash
   kaggle datasets download -d emmarex/plantdisease
   unzip plantdisease.zip -d data/
(Nota: Si el archivo comprimido genera una subcarpeta anidada redundante data/PlantVillage/PlantVillage, elimínala para dejar únicamente las 15 carpetas de clase en el nivel raíz de data/PlantVillage/).Estructura esperada de directorios (ImageFolder estándar de PyTorch):Plaintextdata/PlantVillage/

├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato_Bacterial_spot/
├── Tomato_healthy/
└── ... (15 clases en total)

2. Arquitectura del ProyectoPlaintextplant-disease-cv-pytorch/
├── data/                   # Dataset local (ignorado en Git)
├── models/
│   ├── class_names.json    # Mapeo índice -> nombre de clase
│   ├── confusion_matrix.png# Evaluación visual sobre el conjunto de prueba
│   ├── best_plant_model.pth# Pesos entrenados en PyTorch (ignorado si excede límite)
│   └── plant_disease_model.onnx # Modelo exportado para inferencia productiva
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Splits estratificados, Data Augmentation y pesos de clase
│   ├── model.py            # EfficientNet-B0 con cabeza personalizada y descongelamiento gradual
│   ├── train.py            # Ciclo de entrenamiento en 2 fases con AMP y Cosine Annealing
│   ├── evaluate.py         # Reporte de clasificación, matriz de confusión y Grad-CAM
│   └── export_onnx.py      # Exportación y verificación numérica en ONNX Runtime
├── app.py                  # Tablero interactivo en Streamlit con Grad-CAM en tiempo real
├── requirements.txt        # Dependencias reproducibles
├── .gitignore
└── README.md

3. Decisiones Técnicas y Fundamentos (El Porqué detrás del Código)Cada componente del pipeline responde a un criterio específico de ingeniería de Deep Learning y Visión por Computadora:

A. Elección del Backbone: 

EfficientNet-B0Eficiencia paramétrica: Con solo 5.3 millones de parámetros (frente a los 25M de ResNet50 o 138M de VGG16), EfficientNet-B0 utiliza escalado compuesto (Compound Scaling) que balancea profundidad, anchura y resolución de entrada (224x224).

Bloques MBConv + Squeeze-and-Excitation (SE): Los módulos de atención por canal ponderan dinámicamente los mapas de características más informativos, permitiendo capturar texturas finas como clorosis, necrosis marginal o esporulación fúngica.

B. Entrenamiento en Dos Fases (Feature Extraction → Fine-Tuning)

Fase 1 — Entrenamiento de la Cabeza Clasificadora (Backbone congelado): Al añadir capas lineales inicializadas aleatoriamente sobre un extractor preentrenado en ImageNet, los gradientes iniciales son grandes y ruidosos. Congelar el backbone evita destruir los detectores de bordes y texturas de bajo nivel mientras se estabiliza la nueva cabeza (Linear -> BatchNorm1d -> SiLU -> Dropout -> Linear).

Fase 2 — Fine-Tuning Selectivo con Discriminative Learning Rates: Se descongelan únicamente los bloques convolucionales profundos (a partir del bloque 5), donde se codifican patrones semánticos complejos, y se entrena con tasas de aprendizaje diferenciadas: un Learning Rate 5 veces menor para el backbone (2e-5) que para la cabeza (1e-4). Esto evita el olvido catastrófico (catastrophic forgetting).

C. Manejo de Desbalanceo y Calibración de Confianza:

Split Estratificado (70% Train / 15% Val / 15% Test): Garantiza que todas las enfermedades mantengan exactamente la misma proporción en cada partición, evitando sesgos en la evaluación.

Weighted Cross-Entropy Loss: Dado que PlantVillage presenta un desbalanceo marcado entre clases (algunas superan las 3,000 imágenes y otras rondan las 150), se asignan pesos inversamente proporcionales a la frecuencia de cada clase ($W_c = \frac{N}{C \cdot N_c}$), penalizando más los errores en patologías minoritarias.

Label Smoothing (0.1): Suaviza las etiquetas one-hot (1.0 -> 0.9), evitando que la red produzca logits extremos y mejorando la calibración de probabilidades en inferencia.

D. Data Augmentation Específico del Dominio Agrícola

Invarianza geométrica (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(30°)): A diferencia de objetos con orientación fija (como automóviles o rostros), una hoja puede fotografiarse desde cualquier ángulo.

Robustez fotométrica (ColorJitter): Simula variaciones en iluminación solar, nubosidad y balance de blancos en campo abierto.

Simulación de oclusiones (RandomErasing): Oculta parches rectangulares aleatorios durante el entrenamiento para obligar a la red a distribuir su atención a lo largo de toda la lámina foliar en lugar de memorizar un único punto.

E. Optimización Numérica y Estabilidad del Entrenamiento

Optimizador AdamW: Desacopla el weight decay de la actualización adaptativa del gradiente, logrando mejor regularización L2 que el Adam clásico.

Cosine Annealing LR Scheduler: Reduce la tasa de aprendizaje siguiendo una curva coseno suave, facilitando la convergencia hacia mínimos más planos y generalizables.

Gradient Clipping (max_norm=1.0) y AMP: Previene la explosión de gradientes durante el fine-tuning y habilita precisión mixta automática (FP16/FP32) cuando se dispone de GPU CUDA.

F. Explicabilidad Visual (XAI) mediante Grad-CAM Nativo

Se implementó Grad-CAM (Gradient-weighted Class Activation Mapping) registrando Forward y Backward Hooks directamente sobre el último bloque convolucional (model.backbone.features[-1]). Esto extrae los gradientes de la clase predicha respecto a los mapas de activación espaciales, generando un mapa de calor que audita visualmente que el modelo fundamenta su diagnóstico en las lesiones de la hoja y no en artefactos del fondo (shortcut learning).

G. Despliegue y Producción con ONNXEl script src/export_onnx.py convierte el grafo computacional de PyTorch al estándar abierto ONNX (Open Neural Network Exchange) con ejes dinámicos (dynamic_axes para soportar tamaños de batch variables) y ejecuta una validación numérica (np.testing.assert_allclose) comprobando que las salidas de ONNX Runtime coincidan con PyTorch dentro de una tolerancia de 1e-5.

4. Resultados de Evaluación. Tras completar las dos fases de entrenamiento sobre el subconjunto de 15 clases, el modelo alcanzó ~95% de Accuracy en validación y prueba, demostrando alta capacidad de generalización entre patologías visuales cercanas.

5. Guía Paso a Paso para Ejecutar el Proyecto

Paso 1: Clonar el repositorio e instalar dependencias

Bash git clone [https://github.com/GuillermoA86/plant-disease-cv-pytorch.git](https://github.com/GuillermoA86/plant-disease-cv-pytorch.git)
cd plant-disease-cv-pytorch
python -m venv venv

# En Linux/macOS:
source venv/bin/activate
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

Paso 2: Entrenar el modelo

Bash python -m src.train
Ejecuta el entrenamiento en dos fases, aplica Early Stopping y guarda el mejor checkpoint en models/best_plant_model.pth junto con models/class_names.json.

Paso 3: Evaluar métricas en el conjunto de prueba (Test)

Bash python -m src.evaluate

Imprime en consola el reporte detallado (Precision, Recall, F1-Score por clase) y guarda la matriz de confusión en models/confusion_matrix.png.

Paso 4: Exportar y verificar el modelo en ONNX

Bash python -m src.export_onnx

Genera models/plant_disease_model.onnx y valida su equivalencia numérica frente a PyTorch usando onnxruntime.

Paso 5: Ejecutar el tablero interactivo en Streamlit

Bash streamlit run app.py

Abre la interfaz web local donde podrás subir fotografías de hojas, visualizar el diagnóstico con su porcentaje de confianza, explorar el Top-5 de probabilidades e inspeccionar en tiempo real el mapa de atención Grad-CAM.

6. Limitaciones Conocidas y Trabajo Futuro

Brecha de Dominio (Domain Gap): Las imágenes de PlantVillage fueron capturadas en condiciones controladas sobre fondos uniformes. Para escalar este sistema a fotografías tomadas directamente en campo abierto (in-the-wild), los siguientes pasos naturales son:Incorporar una etapa previa de detección/segmentación de hojas con YOLOv8 o Segment Anything (SAM) para aislar el follaje del fondo complejo.Aplicar Domain Adaptation entrenando con datasets de campo real como PlantDoc.

Cuantización INT8 para Edge AI: Aplicar cuantización post-entrenamiento en ONNX Runtime para desplegar el modelo en dispositivos móviles agrícolas sin conexión a internet.