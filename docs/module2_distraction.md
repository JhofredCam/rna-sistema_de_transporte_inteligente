# Modulo 2: Clasificacion de Conduccion Distractiva

Este modulo entrena y evalua un clasificador de imagenes para detectar comportamientos de conduccion distractiva en vehiculos de transporte.

## Dataset

Dataset recomendado:

https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset/data

Descarga:

```bash
python scripts/download_data.py --output-dir data/raw
```

El loader usa `torchvision.datasets.ImageFolder`. Si encuentra carpetas `train`, `val` o `test`, las respeta. Si no existen, crea divisiones reproducibles de entrenamiento, validacion y prueba.

## Entrenamiento

```bash
python scripts/train_module2_distraction.py ^
  --data-dir data/raw/module2_distraction ^
  --output-dir models/module2_distraction ^
  --architecture resnet18 ^
  --epochs 10 ^
  --batch-size 32
```

Archivos generados:

- `models/module2_distraction/best_model.pth`: checkpoint PyTorch con pesos y metadatos.
- `models/module2_distraction/history.csv`: evolucion de perdida y metricas en validacion.
- `models/module2_distraction/evaluation/metrics.json`: accuracy, precision, recall, F1-score, matriz de confusion y reporte por clase.
- `models/module2_distraction/evaluation/classification_report.csv`: reporte tabular por clase.
- `models/module2_distraction/evaluation/examples/`: ejemplos correctos e incorrectos copiados desde el conjunto de prueba.

## Evaluacion

```bash
python scripts/evaluate_module2_distraction.py ^
  --data-dir data/raw/module2_distraction ^
  --checkpoint models/module2_distraction/best_model.pth
```

Metricas obligatorias del entregable:

- `accuracy`
- `precision`
- `recall`
- `f1_score`

Tambien se guardan versiones macro para detectar si el modelo esta favoreciendo clases frecuentes.

## Inferencia

```bash
python scripts/predict_module2_distraction.py ^
  --checkpoint models/module2_distraction/best_model.pth ^
  --images ruta/a/imagen1.jpg ruta/a/imagen2.jpg
```

La salida incluye la clase predicha, confianza, probabilidades por clase y una medida preventiva sugerida.

## Informe operativo

Para reportar distracciones frecuentes, usar el `classification_report.csv`, la matriz de confusion y los resultados de inferencia sobre imagenes nuevas. Las clases con mayor soporte o mayor recurrencia en predicciones reales deben priorizarse para intervencion:

- Uso del celular o mensajeria: politica de cero uso en marcha, monitoreo de cabina y recordatorios en inicio de turno.
- Somnolencia o fatiga: revision de turnos, pausas activas, alertas tempranas y relevos en rutas largas.
- Manipulacion de radio o dispositivos: configuracion antes de iniciar ruta y bloqueo de ajustes durante conduccion.
- Alcanzar objetos o distraerse con pasajeros: asegurar objetos antes de salida y reforzar protocolos de atencion al pasajero.

## Limitaciones

El repositorio no versiona imagenes ni pesos por `.gitignore`. Los resultados finales deben generarse localmente despues de descargar el dataset con credenciales de Kaggle.
