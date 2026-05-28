# Sistema de Transporte Inteligente

Repositorio para los modulos de analitica y redes neuronales aplicados a una empresa de transporte.

## Modulo 2: Clasificacion de conduccion distractiva

El modulo 2 implementa una linea base en PyTorch para clasificar imagenes de comportamiento del conductor con el dataset [Multi-Class Driver Behavior Image Dataset](https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset).

Clases esperadas:

- `Safe Driving`
- `Turning`
- `Texting Phone`
- `Talking Phones`
- `Others`

El entrenamiento real esta pensado para Google Colab con GPU. Localmente se pueden ejecutar pruebas unitarias sin descargar el dataset completo.

## Estructura esperada del dataset

Despues de descargar y extraer el dataset, la carpeta debe verse como un `ImageFolder` de torchvision:

```text
data/raw/multi_class_driver_behavior/
  Safe Driving/
  Turning/
  Texting Phone/
  Talking Phones/
  Others/
```

Si el zip agrega una carpeta intermedia, el loader intenta encontrar automaticamente la raiz compatible.

## Uso en Colab

1. Abre `notebooks/02_train_distraction_pytorch_colab.ipynb`.
2. Activa GPU en Colab.
3. Sube tu `kaggle.json` o configura `KAGGLE_USERNAME` y `KAGGLE_KEY`.
4. Ejecuta las celdas para descargar datos, entrenar, evaluar y exportar artefactos.

Artefactos principales:

- `models/distraction/best_model.pt`
- `models/distraction/artifacts/metrics.json`
- `models/distraction/artifacts/classification_report.csv`
- `models/distraction/artifacts/confusion_matrix.csv`
- `models/distraction/artifacts/correct_predictions.png`
- `models/distraction/artifacts/error_cases.png`

## Pruebas locales

```bash
pip install -r requirements.txt
pytest tests/unit
```

Las pruebas usan imagenes sinteticas temporales y no descargan el dataset real.
