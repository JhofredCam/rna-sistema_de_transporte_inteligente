# Informe del Modulo 2: Clasificacion de Conduccion Distractiva

## Resumen

Se entreno un modelo de vision por computador para clasificar comportamientos de conduccion a partir de imagenes de cabina. El entrenamiento se realizo con PyTorch usando transferencia de aprendizaje sobre `mobilenet_v3_small`, una arquitectura ligera apropiada para entrenamiento en laptop con GPU de entrada.

Comando usado:

```powershell
python scripts/train_module2_distraction.py --data-dir data/raw/module2_distraction --output-dir models/module2_distraction --architecture mobilenet_v3_small --epochs 16 --batch-size 16
```

Artefactos generados:

- Modelo entrenado: `models/module2_distraction/best_model.pth`
- Historial: `models/module2_distraction/history.csv`
- Metricas: `models/module2_distraction/evaluation/metrics.json`
- Reporte por clase: `models/module2_distraction/evaluation/classification_report.csv`
- Ejemplos: `models/module2_distraction/evaluation/examples/`

## Configuracion del Entrenamiento

| Parametro | Valor |
|---|---:|
| Arquitectura | `mobilenet_v3_small` |
| Pesos preentrenados | Si |
| Epocas | 16 |
| Batch size | 16 |
| Tamano de imagen | 224 x 224 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Dispositivo usado | CUDA |
| Semilla | 42 |

Clases detectadas en el dataset:

- `other_activities`
- `safe_driving`
- `talking_phone`
- `texting_phone`
- `turning`

## Resultados Globales en Prueba

El conjunto de prueba tuvo 1091 imagenes. Los resultados globales fueron:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.9478 |
| Precision ponderada | 0.9485 |
| Recall ponderado | 0.9478 |
| F1-score ponderado | 0.9478 |
| Precision macro | 0.9491 |
| Recall macro | 0.9444 |
| F1-score macro | 0.9463 |

El desempeno es alto y relativamente balanceado entre clases. La diferencia pequena entre F1 ponderado y F1 macro indica que el modelo no depende solamente de las clases con mas imagenes.

## Resultados por Clase

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| `other_activities` | 0.9244 | 0.8785 | 0.9008 | 181 |
| `safe_driving` | 0.8935 | 0.9438 | 0.9180 | 249 |
| `talking_phone` | 0.9911 | 0.9696 | 0.9802 | 230 |
| `texting_phone` | 0.9630 | 0.9915 | 0.9770 | 236 |
| `turning` | 0.9734 | 0.9385 | 0.9556 | 195 |

Las mejores clases fueron `talking_phone` y `texting_phone`, ambas relevantes para detectar uso del celular. La clase mas debil fue `other_activities`, probablemente porque agrupa comportamientos heterogeneos que pueden parecerse a conduccion segura o giro.

## Evolucion del Entrenamiento

El mejor F1-score de validacion fue 0.9616 en la epoca 15. La epoca 16 mantuvo un rendimiento cercano, con F1 de validacion de 0.9588.

| Epoca | Train loss | Val loss | Val accuracy | Val F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.9946 | 0.4085 | 0.8598 | 0.8556 |
| 4 | 0.2156 | 0.1990 | 0.9148 | 0.9160 |
| 8 | 0.1272 | 0.1411 | 0.9496 | 0.9494 |
| 11 | 0.0920 | 0.1354 | 0.9505 | 0.9505 |
| 15 | 0.0517 | 0.1275 | 0.9615 | 0.9616 |
| 16 | 0.0462 | 0.1328 | 0.9588 | 0.9588 |

## Matriz de Confusion

Orden de clases: `other_activities`, `safe_driving`, `talking_phone`, `texting_phone`, `turning`.

| Real \ Predicha | other | safe | talking | texting | turning |
|---|---:|---:|---:|---:|---:|
| other_activities | 159 | 16 | 2 | 2 | 2 |
| safe_driving | 8 | 235 | 0 | 3 | 3 |
| talking_phone | 1 | 3 | 223 | 3 | 0 |
| texting_phone | 2 | 0 | 0 | 234 | 0 |
| turning | 2 | 9 | 0 | 1 | 183 |

Los errores mas visibles ocurren entre `other_activities` y `safe_driving`, y algunos casos de `turning` clasificados como `safe_driving`. Esto tiene sentido operativo porque ciertas posturas del conductor pueden ser ambiguas si la imagen no muestra claramente el gesto.

## Ejemplos Clasificados Correctamente

Ejemplos copiados por el evaluador:

| Imagen | Clase real | Prediccion | Confianza |
|---|---|---|---:|
| `models/module2_distraction/evaluation/examples/correct/00000_img_74433.jpg` | `turning` | `turning` | 1.0000 |
| `models/module2_distraction/evaluation/examples/correct/00002_img_33994.jpg` | `texting_phone` | `texting_phone` | 1.0000 |
| `models/module2_distraction/evaluation/examples/correct/00008_img_33898.jpg` | `talking_phone` | `talking_phone` | 0.9998 |
| `models/module2_distraction/evaluation/examples/correct/00009_img_70040.jpg` | `safe_driving` | `safe_driving` | 0.9996 |
| `models/module2_distraction/evaluation/examples/correct/00013_img_23443.jpg` | `other_activities` | `other_activities` | 0.9998 |

## Casos Erroneos

| Imagen | Clase real | Prediccion | Confianza |
|---|---|---|---:|
| `models/module2_distraction/evaluation/examples/incorrect/00003_IMG_3748.JPG` | `talking_phone` | `safe_driving` | 0.6713 |
| `models/module2_distraction/evaluation/examples/incorrect/00012_IMG_20240930_135811116_HDR_AE.jpg` | `talking_phone` | `texting_phone` | 0.3480 |
| `models/module2_distraction/evaluation/examples/incorrect/00017_img_66097.jpg` | `safe_driving` | `other_activities` | 0.7936 |
| `models/module2_distraction/evaluation/examples/incorrect/00028_IMG_20240930_140125456_HDR_AE.jpg` | `texting_phone` | `other_activities` | 0.9925 |
| `models/module2_distraction/evaluation/examples/incorrect/00094_IMG_20240930_133013841_HDR_AE.jpg` | `turning` | `safe_driving` | 0.5465 |

Estos errores sugieren revisar visualmente los casos ambiguos y, si se desea mejorar el modelo, aumentar imagenes en condiciones reales de operacion: diferentes camaras, iluminacion, posiciones de manos y angulos de cabina.

## Distracciones mas Frecuentes

En el conjunto de prueba, excluyendo `safe_driving`, las clases con mayor soporte fueron:

| Tipo | Imagenes en prueba | Interpretacion |
|---|---:|---|
| `texting_phone` | 236 | Uso del celular para escribir o manipular pantalla |
| `talking_phone` | 230 | Uso del celular para llamada |
| `turning` | 195 | Giro o desviacion de atencion/postura |
| `other_activities` | 181 | Actividades no clasificadas explicitamente |

El uso del telefono aparece como la distraccion principal si se suman `talking_phone` y `texting_phone`.

## Medidas Preventivas

- Uso de telefono: politica de cero uso del celular durante la conduccion, soporte de monitoreo en cabina y sanciones progresivas.
- Mensajeria: bloqueo operativo de mensajeria durante ruta, sensibilizacion sobre tiempos de reaccion y alertas en cabina.
- Giros o cambios de postura: capacitacion sobre preparacion previa de ruta y espejos antes de iniciar marcha.
- Otras actividades: asegurar objetos antes de salir, evitar manipulacion de elementos dentro de cabina y reforzar protocolos de atencion al pasajero.
- Seguimiento operativo: revisar periodicamente los falsos negativos de `talking_phone` y `texting_phone`, ya que son los casos con mayor riesgo vial.

## Conclusiones

El modelo entrenado cumple el objetivo del modulo: clasifica comportamientos de conduccion distractiva con accuracy de 94.78% y F1-score ponderado de 94.78% en prueba. La deteccion de uso del telefono es especialmente fuerte, con F1 de 98.02% para `talking_phone` y 97.70% para `texting_phone`.

Para una siguiente iteracion, se recomienda recolectar mas datos propios de la empresa de transporte, especialmente en condiciones reales de iluminacion, camara, uniforme, asiento y rutas. Tambien conviene revisar manualmente los errores entre `other_activities` y `safe_driving`, porque esa frontera es la mas ambigua del modelo actual.
