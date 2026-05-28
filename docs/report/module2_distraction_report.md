# Informe del modulo 2: Clasificacion de conduccion distractiva

## Objetivo

Entrenar y evaluar un modelo de clasificacion de imagenes para detectar comportamientos de conduccion distractiva en vehiculos de transporte.

## Dataset

Dataset: Multi-Class Driver Behavior Image Dataset.

Clases:

- Safe Driving
- Turning
- Texting Phone
- Talking Phones
- Others

## Modelo

Baseline: `resnet18` preentrenada con ImageNet y cabeza final ajustada al numero de clases. El entrenamiento se realiza en Google Colab con GPU.

## Metricas de evaluacion

El notebook genera:

- Accuracy
- Precision macro y weighted
- Recall macro y weighted
- F1-score macro y weighted
- Matriz de confusion
- Reporte por clase

Los resultados se guardan en `models/distraction/artifacts/`.

## Ejemplos clasificados

El notebook exporta dos grillas:

- `correct_predictions.png`: ejemplos correctamente clasificados.
- `error_cases.png`: casos erroneos con etiqueta real, prediccion y confianza.

## Distracciones frecuentes

El analisis de frecuencia se calcula sobre las predicciones del conjunto de prueba. Las clases de riesgo son:

- Turning
- Texting Phone
- Talking Phones
- Others

Despues de ejecutar el notebook, ordenar `predicted_count` permite identificar los comportamientos distractores mas frecuentes.

## Medidas preventivas sugeridas

- Para `Texting Phone`: politicas de cero uso del movil durante operacion, soportes sellados o bloqueo operativo de apps.
- Para `Talking Phones`: manos libres no debe reemplazar la politica de atencion; priorizar paradas seguras para llamadas.
- Para `Others`: revisar subcasos manualmente, ya que agrupa somnolencia, beber, hablar con pasajeros u otras acciones.
- Para `Turning`: usarlo como senal contextual, no necesariamente como infraccion aislada; puede indicar maniobra normal si no coexiste con otra distraccion.

## Limitaciones

El dataset fue capturado en un contexto geografico y vehicular especifico. Antes de usar el modelo en produccion, se debe validar con imagenes propias de la flota, camaras reales y condiciones de iluminacion locales.
