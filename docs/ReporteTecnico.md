# Reporte Técnico: Sistema Inteligente Integrado para Predicción, Clasificación y Recomendación en una Empresa de Transporte

**Institución:** Universidad Nacional de Colombia  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Proyecto:** Sistema de transporte inteligente basado en aprendizaje profundo  
**Integrantes:** Completar con los nombres del equipo  
**Fecha:** Mayo de 2026  
**Repositorio:** `rna-sistema_de_transporte_inteligente`

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un sistema inteligente para una empresa de transporte que debe tomar mejores decisiones operativas, reducir riesgos viales y ofrecer una experiencia más personalizada a sus usuarios. La solución integra tres módulos de aprendizaje profundo: predicción de demanda de pasajeros a 30 días, clasificación de comportamientos distractores de conductores en imágenes y recomendación personalizada de destinos de viaje.

El módulo de demanda usa una red LSTM bidireccional con mecanismo de atención y embeddings para ruta y clima. Fue entrenado sobre un dataset sintético de 7.500 registros diarios, distribuidos en cinco rutas entre 2024-01-01 y 2028-02-08. El resultado global fue RMSE de 175,83 pasajeros, MAE de 125,86 pasajeros y MAPE de 7,77%. El módulo de visión por computador usa transferencia de aprendizaje con `mobilenet_v3_small` sobre imágenes de cabina; obtuvo accuracy de 94,78% y F1-score ponderado de 94,78% en 1.091 imágenes de prueba. El módulo de recomendación implementa un recomendador neuronal híbrido con embeddings de usuario, embeddings de destino y variables de contenido; sobre 114 usuarios evaluados obtuvo recall@10 de 1,0, hit_rate@10 de 1,0 y NDCG@10 de 0,604.

La herramienta web fue construida con React, Vite, Tailwind CSS y componentes de visualización, conectada a una API FastAPI para los módulos de demanda y clasificación. El recomendador está implementado, entrenado y evaluado desde scripts y artefactos del repositorio; su conexión completa a la API/web queda como mejora inmediata. En conjunto, el sistema demuestra la viabilidad técnica de apoyar planeación de flota, seguridad vial y personalización comercial desde un flujo integrado de datos y modelos.

---

## 2. Introducción

Las empresas de transporte enfrentan tres problemas simultáneos: anticipar cuánta demanda tendrán sus rutas, asegurar que los conductores no presenten comportamientos de riesgo y aumentar la satisfacción del usuario mediante sugerencias de viaje relevantes. Estos retos suelen tratarse por separado, pero en la práctica comparten un mismo objetivo: asignar recursos en el momento adecuado, operar de forma segura y mejorar la relación con los pasajeros.

El proyecto aborda el enunciado mediante tres preguntas técnicas:

1. ¿Cómo predecir la demanda de transporte en rutas específicas durante los próximos 30 días?
2. ¿Cómo clasificar imágenes de conductores para identificar comportamientos distractores?
3. ¿Cómo recomendar destinos de viaje personalizados a partir del historial y preferencias de los usuarios?

El objetivo general es desarrollar un sistema inteligente basado en aprendizaje profundo que integre predicción, clasificación y recomendación dentro de una herramienta web. Los alcances implementados incluyen entrenamiento, evaluación, persistencia de artefactos, inferencia por API en los módulos 1 y 2, y una interfaz de usuario para probar las funcionalidades. Las principales limitaciones son el uso de datos sintéticos para demanda, el uso de datasets externos para visión y recomendación, y la integración web parcial del módulo recomendador.

---

## 3. Metodología

### 3.1 Enfoque general

La metodología se organizó con una lógica cercana a CRISP-DM y Design Thinking:

| Fase | Aplicación en el proyecto |
|---|---|
| Comprensión del problema | Se identificaron necesidades de planeación operativa, seguridad vial y personalización de destinos. |
| Comprensión de datos | Se revisaron estructuras tabulares, series temporales, carpetas de imágenes e interacciones usuario-destino. |
| Preparación de datos | Se generaron variables temporales, escalamiento, codificación de categorías, transformaciones de imagen y muestras negativas. |
| Modelamiento | Se entrenaron una LSTM con atención, una CNN por transfer learning y un recomendador neuronal híbrido. |
| Evaluación | Se usaron métricas específicas por tarea: RMSE/MAE/MAPE, accuracy/F1/precisión/recall y métricas top-K. |
| Despliegue | Se construyó API FastAPI y frontend React para visualizar predicciones y probar inferencias. |

### 3.2 Herramientas y tecnologías

| Capa | Tecnologías |
|---|---|
| Modelado | Python, PyTorch, Torchvision, scikit-learn, pandas, NumPy |
| Visualización y EDA | Matplotlib, Seaborn, notebooks Jupyter |
| API | FastAPI, Pydantic, CORS middleware |
| Web | React, Vite, Tailwind CSS, Recharts, lucide-react |
| Persistencia de artefactos | Checkpoints `.pth`, encoders/scalers `.pkl`, reportes `.json`, `.csv` y figuras `.png` |

### 3.3 Criterios de evaluación

Cada módulo se evaluó con métricas alineadas con el tipo de problema:

| Módulo | Tipo de problema | Métricas principales |
|---|---|---|
| Predicción de demanda | Regresión de serie temporal | RMSE, MAE, MAPE |
| Conducción distractiva | Clasificación multiclase de imágenes | Accuracy, precisión, recall, F1-score, matriz de confusión |
| Recomendación | Ranking personalizado top-K | Precisión@K, Recall@K, Hit Rate@K, MAP@K, NDCG@K |

---

## 4. Desarrollo Técnico por Módulo

## 4.1 Módulo 1: Predicción de Demanda de Transporte

### Problema

La empresa necesita anticipar el número de pasajeros por ruta para planear flota, conductores, frecuencia de servicio y estrategias de contingencia. El horizonte solicitado es de 30 días, por lo que el modelo debe capturar patrones recientes, tendencia y estacionalidad semanal/mensual.

### Dataset

El dataset `data/demanda_transporte.csv` fue generado de forma sintética desde `src/module1_demand/data_generator.py`. Contiene 7.500 registros, cinco rutas y 1.500 días por ruta.

| Característica | Valor |
|---|---:|
| Registros totales | 7.500 |
| Rutas | 5 |
| Periodo | 2024-01-01 a 2028-02-08 |
| Pasajeros promedio | 1.507,36 |
| Pasajeros mínimo / máximo | 346 / 4.039 |
| Climas simulados | Soleado, Lluvia, Nublado |

Promedio de pasajeros por ruta:

| Ruta | Registros | Promedio | Minimo | Maximo |
|---|---:|---:|---:|---:|
| Ruta A | 1.500 | 1.446,28 | 585 | 2.836 |
| Ruta B | 1.500 | 2.150,43 | 830 | 4.039 |
| Ruta C | 1.500 | 950,87 | 346 | 1.935 |
| Ruta D | 1.500 | 1.794,14 | 705 | 3.602 |
| Ruta E | 1.500 | 1.195,08 | 418 | 2.795 |

El generador incorpora demanda base por ruta, deriva de tendencia, estacionalidad semanal, estacionalidad mensual, clima con persistencia, festivos, ventanas alrededor de festivos, días de pago, eventos especiales y ruido autorregresivo AR(1). Esto permite simular un comportamiento más realista que una serie completamente limpia.

### Preprocesamiento

El pipeline en `src/module1_demand/preprocessor.py` realiza:

- Ordenamiento por `ruta` y `fecha`.
- Codificación de `ruta` y `clima` con `LabelEncoder`.
- División temporal por ruta con 80% para entrenamiento y 20% para prueba.
- Escalamiento `MinMaxScaler` de variables temporales (`dia_semana`, `mes`, `festivo`) y variable objetivo (`pasajeros`) sin fuga de información desde test.
- Construcción de ventanas deslizantes de 30 días mediante `build_sequences`.

### Diseno del modelo

El modelo `TransportLSTM` se implementó en `src/module1_demand/model.py`. Su arquitectura combina:

- LSTM bidireccional de 2 capas.
- Tamaño oculto de 160 unidades.
- `LayerNorm` en la entrada.
- Embeddings para ruta y clima.
- Atención temporal para ponderar los días más relevantes dentro de la ventana de 30 días.
- Cabeza densa con activaciones GELU y dropout.

El entrenamiento usa `SmoothL1Loss`, optimizador `AdamW`, scheduler `ReduceLROnPlateau`, clipping de gradientes y early stopping. La inferencia a 30 días se realiza de forma autorregresiva: cada predicción alimenta el contexto del siguiente día.

### Evaluación y resultados

Métricas globales:

| Métrica | Valor |
|---|---:|
| RMSE | 175,83 pasajeros |
| MAE | 125,86 pasajeros |
| MAPE | 7,77% |

Métricas por ruta:

| Ruta | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Ruta A | 156,15 | 116,03 | 7,53% |
| Ruta B | 241,24 | 181,25 | 7,65% |
| Ruta C | 108,40 | 83,37 | 8,21% |
| Ruta D | 196,47 | 143,69 | 7,23% |
| Ruta E | 147,15 | 104,97 | 8,22% |

La Ruta B presenta el mayor RMSE absoluto, consistente con su mayor demanda promedio y mayor rango de pasajeros. La Ruta C tiene el menor RMSE, aunque su MAPE es similar al resto porque opera con menor volumen. En términos relativos, el MAPE se mantiene entre 7,23% y 8,22%, lo que indica estabilidad del modelo entre rutas.

Figuras generadas:

- Predicción vs demanda real: `../models/demand/prediccion_vs_real_por_ruta.png`
- Comparativa de métricas: `../models/demand/comparativa_metricas_por_ruta.png`
- Heatmap de error absoluto: `../models/demand/heatmap_error_por_ruta.png`
- Curva de aprendizaje: `../models/demand/curva_aprendizaje.png`

### Análisis de estacionalidad y tendencia

El comportamiento de la demanda muestra tres efectos principales. Primero, la estacionalidad semanal aumenta la demanda entre lunes y viernes y la reduce durante fines de semana. Segundo, la estacionalidad mensual eleva la demanda en junio, julio y diciembre, meses asociados a vacaciones y desplazamientos especiales. Tercero, la tendencia por ruta introduce crecimiento gradual, con mayor intensidad en rutas de demanda base alta como Ruta B y Ruta D. El clima también afecta la demanda: la lluvia reduce pasajeros, especialmente si ocurre en días consecutivos o durante fines de semana.

Desde el punto de vista operativo, estas señales permiten ajustar frecuencia de buses, reservas de conductores y mantenimiento preventivo antes de picos previsibles.

---

## 4.2 Módulo 2: Clasificación de Conducción Distractiva

### Problema

La empresa requiere detectar comportamientos que puedan afectar la seguridad vial. El sistema clasifica imágenes de cabina para identificar conducción segura o distractores como uso del teléfono, manipulación de dispositivos o desviación de atención.

### Dataset

El dataset recomendado y usado por el flujo del repositorio es `Multi-Class Driver Behavior Image Dataset` disponible en Kaggle. La descarga se automatiza con:

```bash
python scripts/download_data.py --module module2 --output-dir data/raw
```

El cargador usa `torchvision.datasets.ImageFolder`, por lo que cada carpeta representa una clase. Si el dataset no trae divisiones `train`, `val` y `test`, el módulo genera particiones reproducibles con semilla.

Clases detectadas durante el entrenamiento:

- `other_activities`
- `safe_driving`
- `talking_phone`
- `texting_phone`
- `turning`

Una limitación importante es que, aunque el enunciado menciona somnolencia, el dataset local usado no incluye una clase explícita de somnolencia. Por eso el modelo entrenado cubre las cinco clases anteriores.

### Preprocesamiento y aumento de datos

El módulo aplica transformaciones de imagen con Torchvision: redimensionamiento a 224 x 224, normalización y aumentos para entrenamiento. El objetivo del aumento de datos es mejorar la robustez ante variaciones de iluminación, ángulo de cámara, postura y resolución.

### Diseno del modelo

Se implementó transferencia de aprendizaje con `mobilenet_v3_small`, pesos preentrenados y ajuste fino del modelo completo. Esta arquitectura fue elegida por su equilibrio entre rendimiento y costo computacional, especialmente para entrenamiento local con GPU de entrada. El entrenamiento usó:

| Parametro | Valor |
|---|---:|
| Arquitectura | `mobilenet_v3_small` |
| Pesos preentrenados | Si |
| Epocas | 16 |
| Batch size | 16 |
| Tamaño de imagen | 224 x 224 |
| Learning rate | 0,0001 |
| Weight decay | 0,0001 |
| Semilla | 42 |
| Dispositivo | CUDA |

### Evaluación y resultados

El conjunto de prueba contiene 1.091 imágenes.

| Métrica | Valor |
|---|---:|
| Accuracy | 0,9478 |
| Precisión ponderada | 0,9485 |
| Recall ponderado | 0,9478 |
| F1-score ponderado | 0,9478 |
| Precisión macro | 0,9491 |
| Recall macro | 0,9444 |
| F1-score macro | 0,9463 |

Resultados por clase:

| Clase | Precisión | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| `other_activities` | 0,9244 | 0,8785 | 0,9008 | 181 |
| `safe_driving` | 0,8935 | 0,9438 | 0,9180 | 249 |
| `talking_phone` | 0,9911 | 0,9696 | 0,9802 | 230 |
| `texting_phone` | 0,9630 | 0,9915 | 0,9770 | 236 |
| `turning` | 0,9734 | 0,9385 | 0,9556 | 195 |

Matriz de confusión:

| Real \ Predicha | other | safe | talking | texting | turning |
|---|---:|---:|---:|---:|---:|
| other_activities | 159 | 16 | 2 | 2 | 2 |
| safe_driving | 8 | 235 | 0 | 3 | 3 |
| talking_phone | 1 | 3 | 223 | 3 | 0 |
| texting_phone | 2 | 0 | 0 | 234 | 0 |
| turning | 2 | 9 | 0 | 1 | 183 |

El modelo funciona especialmente bien para `talking_phone` y `texting_phone`, las clases más críticas para intervención por uso de celular. La clase más compleja es `other_activities`, porque agrupa comportamientos heterogéneos que pueden parecerse a conducción segura.

### Ejemplos correctos y erroneos

El evaluador guarda evidencia visual en `models/module2_distraction/evaluation/examples/` y resume los casos en `examples.json`.

Ejemplos correctos:

| Imagen | Real | Predicción | Confianza |
|---|---|---|---:|
| `00000_img_74433.jpg` | `turning` | `turning` | 1,0000 |
| `00002_img_33994.jpg` | `texting_phone` | `texting_phone` | 1,0000 |
| `00008_img_33898.jpg` | `talking_phone` | `talking_phone` | 0,9998 |
| `00009_img_70040.jpg` | `safe_driving` | `safe_driving` | 0,9996 |

Casos erroneos:

| Imagen | Real | Predicción | Confianza |
|---|---|---|---:|
| `00003_IMG_3748.JPG` | `talking_phone` | `safe_driving` | 0,6713 |
| `00012_IMG_20240930_135811116_HDR_AE.jpg` | `talking_phone` | `texting_phone` | 0,3480 |
| `00017_img_66097.jpg` | `safe_driving` | `other_activities` | 0,7936 |
| `00028_IMG_20240930_140125456_HDR_AE.jpg` | `texting_phone` | `other_activities` | 0,9925 |

### Distracciones frecuentes y medidas preventivas

En el conjunto de prueba, excluyendo `safe_driving`, las clases con mayor soporte fueron `texting_phone` (236), `talking_phone` (230), `turning` (195) y `other_activities` (181). Al sumar `talking_phone` y `texting_phone`, el uso de celular aparece como la distracción dominante.

Medidas sugeridas:

- Uso de teléfono: política de cero uso durante conducción, alertas en cabina y sanciones progresivas.
- Mensajería: bloqueo operativo o modo conducción en dispositivos corporativos.
- Giros o desviación de atención: capacitación sobre preparación previa de ruta, espejos y elementos de cabina.
- Otras actividades: asegurar objetos antes de iniciar viaje y reforzar protocolos de atención al pasajero.
- Monitoreo: revisar falsos negativos de uso de celular, porque son los casos de mayor riesgo.

---

## 4.3 Módulo 3: Sistema de Recomendación de Destinos de Viaje

### Problema

El sistema debe sugerir destinos personalizados para usuarios de la empresa de transporte con base en interacciones históricas, preferencias previas y atributos de destino. En una plataforma de reservas, esto puede aumentar conversión, retención y descubrimiento de rutas.

### Dataset

El dataset recomendado es `Travel Recommendation Dataset` de Kaggle. Se descarga con:

```bash
python scripts/download_data.py --module module3 --output-dir data/raw
```

Los archivos fuente usados por el entrenamiento fueron:

- `Expanded_Destinations.csv`
- `Final_Updated_Expanded_Reviews.csv`
- `Final_Updated_Expanded_UserHistory.csv`
- `Final_Updated_Expanded_Users.csv`

El cargador infiere alias comunes de columnas para usuario, destino, calificación y tiempo. Cuando no existe rating explícito, la interacción se interpreta como feedback implícito positivo.

### Preprocesamiento

El pipeline construye interacciones usuario-destino, genera muestras negativas por usuario y conserva metadata de contenido. La division train/valid/test se realiza por usuario, de modo que el modelo evalúa si puede recuperar destinos retenidos a partir de un historial parcial.

Una decisión importante fue evaluar por nombre de destino cuando existe metadata `Name`, porque el dataset contiene varios `DestinationID` para un mismo lugar turístico. Así se evita penalizar como error una recomendación equivalente con otro identificador.

### Diseno del modelo

El modelo `HybridTravelRecommender` combina:

- Embeddings de usuario.
- Embeddings de destino.
- Features de contenido del destino.
- Red densa con capas ocultas de 128 y 64 unidades.
- Dropout de 0,2.
- Función objetivo `BCEWithLogitsLoss`.
- Muestreo negativo con 4 negativos por positivo.

Configuración principal:

| Parametro | Valor |
|---|---:|
| Usuarios | 675 |
| Items / destinos | 684 |
| Dimensión de contenido | 21 |
| Embedding dim | 64 |
| Hidden dim | 128 |
| Batch size | 256 |
| Learning rate | 0,001 |
| Weight decay | 0,00001 |
| Paciencia | 5 |

### Evaluación y resultados

| Métrica | Valor |
|---|---:|
| Precisión@5 | 0,2000 |
| Recall@5 | 1,0000 |
| Hit Rate@5 | 1,0000 |
| MAP@5 | 0,4756 |
| NDCG@5 | 0,6037 |
| Precisión@10 | 0,1000 |
| Recall@10 | 1,0000 |
| Hit Rate@10 | 1,0000 |
| MAP@10 | 0,4756 |
| NDCG@10 | 0,6037 |
| Usuarios evaluados | 114 |
| Interacciones retenidas | 114 |

El recall@10 y hit_rate@10 de 1,0 indican que, para cada usuario evaluado, el destino retenido aparece dentro del top 10. La precisión@10 de 0,1 es esperable porque había una sola interacción relevante retenida por usuario; si una lista de 10 contiene exactamente ese acierto, la precisión es 1/10.

Ejemplos de recomendación:

| Usuario | Destino retenido | Recomendaciones destacadas |
|---|---|---|
| 15 | Kerala Backwaters | Kerala Backwaters, Goa Beaches, Taj Mahal, Leh Ladakh, Jaipur City |
| 20 | Taj Mahal | Kerala Backwaters, Goa Beaches, Taj Mahal, Leh Ladakh, Jaipur City |
| 34 | Leh Ladakh | Kerala Backwaters, Goa Beaches, Taj Mahal, Leh Ladakh, Jaipur City |

### Análisis de efectividad

La recuperación completa del destino retenido sugiere que el recomendador aprende patrones útiles de popularidad, historial y contenido. Sin embargo, los ejemplos muestran repetición de destinos altamente populares, lo que indica riesgo de baja diversidad. Para una versión productiva se recomienda medir cobertura de catálogo, diversidad intra-lista, novedad y sesgo hacia destinos populares.

El sistema todavía no resuelve completamente el arranque frío. Para usuarios nuevos se recomienda combinar popularidad, preferencias declaradas, ubicación, presupuesto y temporada de viaje.

---

## 5. Herramienta Web

### 5.1 Arquitectura

La solución web está organizada en dos capas:

| Capa | Componentes |
|---|---|
| Backend | FastAPI en `api/main.py`, routers `demand.py` y `distraction.py`, carga de modelos en `api/dependencies.py` |
| Frontend | React + Vite en `web/`, componentes `SystemForm`, `ModuleResult`, `Hero`, `Resources`, `ReadmeViewer` |

La API expone:

- `GET /`: estado del sistema.
- `GET /demand/metadata`: metadatos de rutas, clima, escaladores y modelo.
- `POST /demand/predict`: pronóstico de demanda a 1-30 días.
- `GET /distraction/health`: estado del clasificador.
- `GET /distraction/classes`: clases disponibles y medidas preventivas.
- `POST /distraction/predict`: clasificación de imagen subida.

### 5.2 Funcionalidades implementadas

| Funcionalidad requerida | Estado en el repositorio |
|---|---|
| Visualizar predicciones de demanda | Implementada con API y frontend; usa histórico local y pronóstico a 30 días. |
| Subir imagen y ver categoría asignada | Implementada con API FastAPI y clasificador PyTorch. |
| Probar recomendaciones personalizadas | Implementada como módulo entrenado y evaluado en scripts; en frontend existe demo estática. |
| Documentación visible desde web | El frontend incluye visor Markdown desde `web/public/README.md`. |

### 5.3 Observación sobre integración

El recomendador cuenta con entrenamiento, evaluación, checkpoint y CLI (`scripts/recommend_module3_destinations.py`), pero `api/routers/recommender.py` está vacío y no se incluye en `api/main.py`. Por tanto, la recomendación neuronal está resuelta a nivel de modelo, pero requiere una tarea final de integración para exponerla como endpoint real y reemplazar la demo estática del frontend.

---

## 6. Resultados Generales y Discusión

### 6.1 Comparación de resultados

| Módulo | Resultado principal | Lectura operativa |
|---|---|---|
| Demanda | MAPE global de 7,77% | Permite planear capacidad con error relativo moderado y estable entre rutas. |
| Clasificación | F1 ponderado de 94,78% | Viable para detección automática asistida de conductas distractoras. |
| Recomendación | Recall@10 de 1,0 | Recupera preferencias retenidas, aunque debe mejorar diversidad y arranque frío. |

### 6.2 Impacto en la empresa de transporte

El módulo de demanda apoya decisiones de flota, turnos, mantenimiento y frecuencia por ruta. El módulo de clasificación permite priorizar intervenciones de seguridad vial con evidencia visual y medidas preventivas asociadas. El módulo de recomendación puede fortalecer la experiencia de usuario en una plataforma de reservas, similar a un sistema de comercio electrónico de viajes: recomienda destinos, reduce fricción de búsqueda y puede impulsar rutas estratégicas.

### 6.3 Comparación con trabajos previos

La arquitectura LSTM se apoya en la capacidad de las redes recurrentes para capturar dependencias temporales. El uso de atención mejora la lectura del historial reciente al permitir que ciertos días pesen más que otros. Para imágenes, la transferencia de aprendizaje sigue una práctica común en visión por computador: partir de redes entrenadas en grandes datasets y ajustarlas al dominio específico. Para recomendación, el enfoque neuronal híbrido combina filtrado colaborativo con atributos de contenido, una estrategia adecuada cuando existen tanto interacciones históricas como metadata de items.

### 6.4 Limitaciones

- Demanda: el dataset es sintético, por lo que se requiere validación con datos reales de recaudo, ocupación, rutas, clima y eventos.
- Clasificación: las imágenes provienen de un dataset externo; podría existir diferencia de dominio frente a cámaras reales de la empresa.
- Somnolencia: no se entrenó una clase explícita porque el dataset usado no la contiene.
- Recomendación: hay riesgo de sobreexposición de destinos populares y no se resuelve por completo el arranque frío.
- Web: la API productiva del recomendador queda pendiente.

---

## 7. Conclusiones y Recomendaciones

El sistema cumple el objetivo principal de integrar tres soluciones de aprendizaje profundo para transporte inteligente. La predicción de demanda alcanza un error relativo global inferior al 8%, suficiente para un primer prototipo de planeación. El clasificador de conducción distractiva logra alto rendimiento y detecta especialmente bien el uso del teléfono, una de las conductas más relevantes para seguridad vial. El recomendador recupera los destinos retenidos en evaluación top-K, aunque debe fortalecerse con diversidad, explicabilidad y manejo de usuarios nuevos.

Recomendaciones para trabajo futuro:

1. Entrenar el módulo de demanda con históricos reales de validación, recaudo, ocupación y eventos externos.
2. Agregar variables exógenas reales como clima observado, incidentes, calendario local, obras y eventos masivos.
3. Ampliar el dataset de conducción con imágenes propias, condiciones nocturnas y una clase explícita de somnolencia.
4. Incorporar Grad-CAM o mapas de calor para auditar que la CNN observe regiones relevantes de la cabina.
5. Conectar el recomendador a FastAPI y al frontend usando el checkpoint `models/module3_recommender/best_model.pth`.
6. Medir diversidad, cobertura y novedad en las recomendaciones, no solo recall.
7. Implementar monitoreo de drift de datos y reentrenamiento periódico.

---

## 8. Aspectos Éticos y Creatividad

### Privacidad y protección de datos

El sistema puede procesar imágenes de conductores e historiales de viaje, por lo que debe cumplir principios de minimización, consentimiento informado, control de acceso y retención limitada. En producción, las imágenes deberían anonimizarse cuando sea posible, almacenarse cifradas y usarse exclusivamente para fines de seguridad definidos.

### Sesgos y justicia algorítmica

El clasificador podría sesgarse por ángulo de cámara, iluminación, tipo de vehículo, postura, género, tono de piel, uniforme o calidad de imagen. El recomendador puede favorecer destinos populares y reducir exposición de destinos emergentes. El módulo de demanda, si se entrena con datos históricos sesgados, podría perpetuar baja frecuencia en rutas tradicionalmente subatendidas.

### Uso responsable

Las predicciones deben apoyar decisiones humanas, no reemplazarlas sin supervisión. En seguridad vial, una clasificación automática no debería ser la única base para sanciones; se recomienda revisión humana en casos de baja confianza o consecuencias disciplinarias.

### Creatividad de la solución

El proyecto no se limita a entrenar modelos aislados. Integra planeación operativa, seguridad y experiencia de usuario en una misma herramienta web. Además, traduce predicciones en acciones: asignación de recursos para demanda, medidas preventivas para distractores y sugerencias personalizadas para usuarios.

---

## 9. Bibliografía

Arafat Sahin Afridi. (s. f.). *Multi-Class Driver Behavior Image Dataset*. Kaggle. https://www.kaggle.com/datasets/arafatsahinafridi/multi-class-driver-behavior-image-dataset

Aman Mehra. (s. f.). *Travel Recommendation Dataset*. Kaggle. https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset

FastAPI. (s. f.). *FastAPI documentation*. https://fastapi.tiangolo.com/

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778. https://doi.org/10.1109/CVPR.2016.90

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web*, 173-182. https://doi.org/10.1145/3038912.3052569

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1314-1324. https://doi.org/10.1109/ICCV.2019.00140

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library

React. (s. f.). *React documentation*. https://react.dev/

Vite. (s. f.). *Vite documentation*. https://vite.dev/

---

## 10. Anexos

### 10.1 Código fuente principal

| Módulo | Archivos principales |
|---|---|
| Demanda | `src/module1_demand/model.py`, `src/module1_demand/train.py`, `src/module1_demand/preprocessor.py`, `src/module1_demand/evaluator.py`, `src/module1_demand/predictor.py` |
| Conducción distractiva | `src/module2_distraction/model.py`, `src/module2_distraction/trainer.py`, `src/module2_distraction/evaluator.py`, `src/module2_distraction/classifier.py` |
| Recomendación | `src/module3_recommender/model.py`, `src/module3_recommender/trainer.py`, `src/module3_recommender/evaluator.py`, `src/module3_recommender/recommender.py` |
| API | `api/main.py`, `api/dependencies.py`, `api/routers/demand.py`, `api/routers/distraction.py` |
| Web | `web/src/App.jsx`, `web/src/model/transportModel.js`, `web/src/components/SystemForm.jsx`, `web/src/components/ModuleResult.jsx` |

### 10.2 Artefactos generados

| Módulo | Artefactos |
|---|---|
| Demanda | `models/demand/best_model.pth`, `metrics.json`, `metrics_por_ruta.csv`, `predicciones_detalle.csv`, figuras `.png`, scalers y encoders `.pkl` |
| Conducción distractiva | `models/module2_distraction/best_model.pth`, `metadata.json`, `history.csv`, `evaluation/metrics.json`, `classification_report.csv`, `evaluation/examples/examples.json` |
| Recomendación | `models/module3_recommender/best_model.pth`, `metadata.json`, `history.csv`, `evaluation/metrics.json`, `evaluation/examples.json` |

### 10.3 Comandos de reproducción

Entrenamiento de demanda:

```bash
python src/module1_demand/train.py
```

Entrenamiento de conducción distractiva:

```bash
python scripts/train_module2_distraction.py --data-dir data/raw/module2_distraction --output-dir models/module2_distraction --architecture mobilenet_v3_small --epochs 16 --batch-size 16
```

Evaluación de conducción distractiva:

```bash
python scripts/evaluate_module2_distraction.py --data-dir data/raw/module2_distraction --checkpoint models/module2_distraction/best_model.pth
```

Entrenamiento del recomendador:

```bash
python scripts/train_module3_recommender.py --data-dir data/raw/module3_recommender --output-dir models/module3_recommender --epochs 20 --batch-size 256
```

Recomendaciones por usuario:

```bash
python scripts/recommend_module3_destinations.py --checkpoint models/module3_recommender/best_model.pth --user-id 15 --top-k 5
```

### 10.4 Enlaces pendientes para entrega

- URL de despliegue frontend: `https://sistema-transporte-inteligente-rna.netlify.app`.
