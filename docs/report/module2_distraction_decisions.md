# Decisiones tecnicas del modulo 2

## Dataset

Se usa el dataset Multi-Class Driver Behavior Image Dataset de Kaggle. La ficha publica de Zenodo describe cinco clases de comportamiento: `Safe Driving`, `Turning`, `Texting Phone`, `Talking Phones` y `Others`. Las imagenes fueron capturadas en condiciones reales dentro de buses y carros particulares, lo que hace razonable priorizar aumentos de color, encuadre y rotacion moderada.

## Modelo

La linea base usa transfer learning con `resnet18` de `torchvision`. Es una arquitectura suficientemente liviana para entrenar en Google Colab y sirve como referencia reproducible antes de probar modelos mas grandes. El codigo permite cambiar a `resnet34` o `mobilenet_v3_small`.

## Entrenamiento

El entrenamiento se ejecuta en Colab, no localmente. El repositorio contiene el pipeline, pruebas sinteticas y notebook operativo. El checkpoint guarda pesos, clases, tamano de imagen, nombre del backbone, historial y metricas de validacion.

## Splits

Los splits train/validation/test son estratificados por clase con semilla fija. Esto reduce variacion accidental en las metricas y mantiene representacion minima de cada clase cuando hay suficientes imagenes.

## Metricas

Se reportan accuracy, precision, recall y F1 macro/weighted. F1 macro es la metrica principal para seleccion de checkpoint porque penaliza el mal rendimiento en clases minoritarias.

## Prevencion

`Safe Driving` se considera la clase segura. `Turning`, `Texting Phone`, `Talking Phones` y `Others` se reportan como comportamientos de atencion o riesgo para orientar acciones preventivas.
