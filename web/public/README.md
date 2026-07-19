# Informe Técnico: Sistema Inteligente para la Empresa de Transporte

> **⚠️ DOCUMENTO EN CONSTRUCCIÓN**
> Este reporte técnico será completado manualmente. Las secciones siguientes contienen placeholders
> que indican el contenido esperado para cada capítulo del informe.

---

## 1. Introducción y delimitación del problema

<!-- AQUÍ: Resumen ejecutivo del proyecto. Describir el contexto de la empresa de transporte,
el problema a resolver (gestión de demanda, seguridad vial, personalización de rutas) y la
propuesta de solución basada en deep learning. -->

## 2. Análisis descriptivo de datos

<!-- AQUÍ: Descripción del dataset de transporte utilizado. Incluir número de registros,
variables relevantes, distribución de rutas, horarios, tipos de incidentes, perfil de clientes.
Mencionar el proceso de limpieza y selección de variables. -->

### 2.1 Figuras EDA

<!-- COLOQUE AQUÍ LA FIGURA DE CORRELACIONES DEL SISTEMA DE TRANSPORTE -->
<!-- COLOQUE AQUÍ LA FIGURA DE DISTRIBUCIÓN DE DEMANDA POR RUTA -->
<!-- COLOQUE AQUÍ LA FIGURA DE VARIABLES MÁS IMPORTANTES DEL EDA -->

### 2.2 Notebooks EDA del repositorio

El análisis exploratorio completo por módulo está disponible en el repositorio:

- Módulo 1 (demanda): [`notebooks/01_eda_demand.ipynb`](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/notebooks/01_eda_demand.ipynb)
- Módulo 2 (distracción): [`notebooks/02_eda_images.ipynb`](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/notebooks/02_eda_images.ipynb)
- Módulo 3 (recomendación): [`notebooks/03_eda_recommender.ipynb`](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/notebooks/03_eda_recommender.ipynb)

## 3. Metodología de modelamiento

<!-- AQUÍ: Descripción de la metodología empleada para los tres módulos. Incluir
la arquitectura de cada modelo (LSTM para series de tiempo, CNN para clasificación de
imágenes, filtrado colaborativo para recomendación), preprocesamiento, división
train/test, y estrategia de validación. -->

### 3.1 Módulo 1: Predicción de Demanda (Series de Tiempo)

<!-- AQUÍ: Metodología específica del modelo LSTM/Prophet para predicción de demanda
a 30 días. Variables de entrada: demanda histórica, capacidad de rutas, tasa de ocupación,
día de la semana, estacionalidad. -->

### 3.2 Módulo 2: Clasificación de Conducción Distractiva (Visión por Computadora)

<!-- AQUÍ: Metodología del modelo CNN/ResNet para clasificación de imágenes de conductores.
Categorías: Conducción segura, uso de teléfono, somnolencia, comiendo, etc. Técnicas de
data augmentation y transfer learning empleadas. -->

### 3.3 Módulo 3: Sistema de Recomendación de Destinos

<!-- AQUÍ: Metodología del sistema de filtrado colaborativo e híbrido para
recomendación de destinos. Descripción del perfilamiento de usuarios, matriz de
preferencias y algoritmo de recomendación. -->

### 3.4 Preprocesamiento y Baseline

<!-- AQUÍ: Descripción del pipeline de preparación de datos para cada módulo.
Modelo baseline contra el que se compara cada módulo. -->

## 4. Evaluación comparativa de desempeño

<!-- AQUÍ: Tabla comparativa de métricas para los tres módulos:

| Módulo | Métrica 1 | Métrica 2 | Baseline |
|---|---|---|---|
| Demanda | RMSE | MAE | ? |
| Conducción | Accuracy | F1-Score | ? |
| Recomendación | Precision | Recall | ? |

-->

### 4.1 Resultados del Módulo de Demanda (RNA-LSTM)

<!-- AQUÍ: Reporte detallado de resultados de la predicción de demanda. -->

### 4.2 Resultados del Módulo de Clasificación (CNN)

<!-- AQUÍ: Reporte detallado de resultados de clasificación de conductores. -->

### 4.3 Resultados del Módulo de Recomendación

<!-- AQUÍ: Reporte detallado de recomendaciones y precisión del sistema. -->

### 4.4 Interpretación técnica

<!-- AQUÍ: Análisis comparativo de los tres módulos vs sus baselines. Discusión
de fortalezas y debilidades de cada enfoque. -->

### 4.5 Figuras de evaluación

<!-- COLOQUE AQUÍ LAS FIGURAS DE EVALUACIÓN DE CADA MÓDULO -->

## 5. Análisis de importancia e interpretabilidad

<!-- AQUÍ: Análisis de importancia de variables para el módulo de predicción de demanda.
Interpretabilidad de los modelos CNN mediante técnicas como Grad-CAM o LIME.
Interpretación de negocio de los resultados de cada módulo. -->

### 5.1 Importancia exploratoria de variables

<!-- AQUÍ: Ranking de variables más influyentes en la predicción de demanda. -->

### 5.2 Interpretabilidad de los modelos

<!-- AQUÍ: SHAP/LIME para demanda, Grad-CAM para clasificación de imágenes,
análisis de componentes para recomendación. -->

### 5.3 Lectura de negocio

<!-- AQUÍ: Interpretación de los hallazgos en lenguaje de negocio para la
empresa de transporte. -->

### 5.4 Figuras de interpretabilidad

<!-- COLOQUE AQUÍ LAS FIGURAS DE SHAP, GRAD-CAM, ETC. -->

## 6. Aplicación web y despliegue

### 6.1 Arquitectura de la aplicación

<!-- AQUÍ: Descripción de la aplicación web desarrollada con React + Vite + Tailwind CSS.
Arquitectura de componentes, integración con los modelos, y flujo de usuario. -->

### 6.2 Repositorio y despliegue

<!-- AQUÍ: Enlace al repositorio del proyecto y URL de la aplicación desplegada. -->

### 6.3 Figura de aplicación web

<!-- COLOQUE AQUÍ CAPTURA DE PANTALLA DE LA APLICACIÓN WEB -->

## 7. Aprendizajes, conclusiones y limitaciones

<!-- AQUÍ: Conclusiones generales del proyecto. Principales aprendizajes en la
implementación de los tres módulos. Limitaciones identificadas y trabajo futuro. -->

## 8. Referencias

<!-- AQUÍ: Lista de referencias bibliográficas y recursos utilizados en el proyecto. -->

Documentos complementarios del repositorio:

- [Enunciado del proyecto (`description.md`)](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/description.md)
- [Reporte técnico integral (`docs/ReporteTecnico.md`)](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/docs/ReporteTecnico.md)
- [Reflexión ética y sesgos (`docs/ethics/etica_y_sesgos.md`)](https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente/blob/main/docs/ethics/etica_y_sesgos.md)

<!-- Ejemplos de referencias (completar con las fuentes reales):
- Kaggle. (s. f.). *Transportation Dataset*. ...
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. ...
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. ...
- ...
-->
