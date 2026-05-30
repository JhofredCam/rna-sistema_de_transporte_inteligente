# Decisiones de Implementacion del Modulo 2

## Enfoque de modelado

Se implemento un clasificador PyTorch con transfer learning usando `torchvision`. La arquitectura por defecto es `resnet18` porque ofrece buen equilibrio entre costo computacional y desempeno para un proyecto academico, pero el CLI permite `mobilenet_v3_small` y `efficientnet_b0`.

## Estructura del dataset

Se eligio `ImageFolder` como contrato de datos porque el dataset de Kaggle esta organizado por carpetas de clase o puede adaptarse facilmente a ese formato. El loader detecta carpetas `train`, `val` y `test`; si no estan presentes, crea splits reproducibles con semilla.

## Metricas

El entregable pide accuracy, F1-score, precision y recall. Se guardan metricas ponderadas y macro:

- Ponderadas: reflejan el rendimiento global cuando hay desbalance.
- Macro: muestran si una clase minoritaria queda mal clasificada aunque el promedio global sea alto.

## Ejemplos correctos y erroneos

La evaluacion copia muestras clasificadas correctamente y errores a `evaluation/examples`. Esto evita depender del notebook para auditar visualmente los casos y facilita incluir evidencia en el informe.

## Medidas preventivas

Las medidas preventivas se mantienen como reglas simples en `classifier.py`, asociadas a palabras clave de las clases. Esto separa la prediccion tecnica de la accion operativa y permite adaptar nombres de clases del dataset sin reentrenar.

## Archivos transversales

Se agregaron utilidades compartidas para semillas, dispositivo, JSON, clases base y metricas. La intencion es dejar una base reutilizable para los modulos 1 y 3 sin imponer una abstraccion pesada.

## Reproducibilidad

El entrenamiento fija semilla y guarda configuracion, clases, arquitectura, tamano de imagen e historial. El checkpoint contiene metadatos suficientes para cargar el modelo sin conocer de antemano el numero ni el orden de clases.

## Integracion

El modulo puede usarse desde:

- Scripts CLI para entrenamiento, evaluacion e inferencia.
- Importaciones Python desde `src.module2_distraction`.
- API FastAPI cuando exista `models/module2_distraction/best_model.pth`.
