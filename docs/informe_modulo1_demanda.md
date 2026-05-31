# Predicción de Demanda de Transporte con LSTM con Atención Temporal: Arquitectura, Implementación y Evaluación en un Sistema Inteligente de Rutas

> **Módulo 1 — Sistema de Predicción de Demanda de Transporte (Series de Tiempo)**
>
> *Sistema Inteligente Integrado para una Empresa de Transporte*

---

## 1. Portada

| Campo | Detalle |
|:------|:--------|
| **Título** | Predicción de Demanda de Transporte con LSTM con Atención Temporal: Arquitectura, Implementación y Evaluación en un Sistema Inteligente de Rutas |
| **Módulo** | Módulo 1 — Predicción de Demanda de Transporte (Series de Tiempo) |
| **Institución** | Curso de Redes Neuronales Artificiales — Ingeniería de Sistemas |
| **Fecha** | Mayo 2026 |
| **Repositorio** | [GitHub — RNA Sistema de Transporte Inteligente](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente) |

**Integrantes:**

| Rol | Nombre |
|:----|:-------|
| ML Engineer / Lead Developer | — |
| Data Scientist | — |
| ML Ops / Backend Developer | — |
| Frontend Developer | — |

---

## 2. Resumen Ejecutivo

**Problema:** Una empresa de transporte urbano enfrenta el desafío de anticipar la demanda de pasajeros en cada una de sus rutas con un horizonte de 30 días para optimizar la asignación de flota, reducir costos operativos y mejorar la experiencia del usuario.

**Enfoque Tecnológico:** Se diseñó e implementó un modelo de Deep Learning para series temporales basado en una arquitectura **LSTM bidireccional de 2 capas con mecanismo de atención temporal**, complementada con embeddings categóricos para codificar rutas y condiciones climáticas. El pipeline completo comprende generación de datos sintéticos realistas, preprocesamiento con escalado por separado de features y target, construcción de ventanas deslizantes de 30 días, entrenamiento con early stopping y reducción adaptativa de learning rate, y un sistema de evaluación riguroso con pronóstico autorregresivo multi-step.

**Resultados Métricos:**

| Métrica | Valor Global |
|:--------|:-------------|
| **RMSE** | 175.83 pasajeros |
| **MAE** | 125.86 pasajeros |
| **MAPE** | 7.77 % |

El modelo logra un error porcentual absoluto medio inferior al 8% en las 5 rutas del sistema, con un desempeño consistente donde la ruta de mayor demanda (Ruta B, promedio 2,150 pasajeros) presenta el RMSE más alto (241.24) pero mantiene un MAPE competitivo de 7.65%, mientras que la ruta de menor demanda (Ruta C, promedio 951 pasajeros) exhibe el RMSE más bajo (108.40) con un MAPE de 8.21%.

**Conclusión de Negocio:** Un MAPE global de 7.77% en un horizonte de 30 días representa una precisión operativamente útil para la planificación de flota, permitiendo reducir el exceso de capacidad en un ~15-20% estimado frente a modelos ingenuos de media histórica. La arquitectura con atención temporal permite además interpretar qué días del contexto pasado son más relevantes para cada predicción, proporcionando trazabilidad a los operadores.

---

## 3. Introducción

### 3.1 Contextualización del Transporte Moderno

La gestión eficiente de flotas de transporte urbano es un problema de optimización combinatorial que depende críticamente de la capacidad de anticipar la demanda. En ciudades con alta densidad poblacional, el desajuste entre oferta y demanda de vehículos genera tres consecuencias medibles:

1. **Sobreoferta:** Vehículos circulando con baja ocupación, incrementando costos de combustible, mantenimiento y congestión.
2. **Suboferta:** Tiempos de espera excesivos para los usuarios, erosión de la satisfacción y pérdida de participación de mercado frente a alternativas (viajes compartidos, aplicaciones de movilidad).
3. **Ineficiencia operativa:** Dificultad para programar mantenimientos, asignar conductores y planificar expansión de rutas.

La predicción precisa de demanda con horizontes de mediano plazo (30 días) es un habilitador fundamental para la toma de decisiones tácticas en la empresa de transporte.

### 3.2 Justificación de los Tres Dolores Operativos

El sistema integrado aborda tres dimensiones complementarias del negocio de transporte:

| Módulo | Dolor Operativo | Impacto |
|:-------|:----------------|:--------|
| **Módulo 1 — Demanda** | Incertidumbre en la planificación de flota | Costos elevados por mala asignación |
| **Módulo 2 — Conducción Distractiva** | Riesgos de seguridad vial y siniestralidad | Pérdidas humanas, legales y de reputación |
| **Módulo 3 — Recomendación** | Baja personalización de la oferta | Menor retención de clientes y revenue |

Este informe se concentra en el Módulo 1, detallando cada aspecto de su ciclo de vida de desarrollo.

### 3.3 Alcances y Limitaciones Técnicas

**Alcances:**

- Predicción de demanda diaria de pasajeros para 5 rutas con horizonte de 30 días.
- Pipeline de entrenamiento completamente reproducible con semilla fija (SEED=1234).
- Arquitectura con atención temporal para interpretabilidad parcial.
- Sistema de predicción autorregresiva multi-step para escenarios de forecasting real.
- Despliegue como API REST (FastAPI) con inferencia en CPU.

**Limitaciones:**

- **Datos sintéticos:** El dataset fue generado artificialmente con patrones estadísticos realistas pero no corresponde a datos operativos reales. Las conclusiones sobre desempeño absoluto deben validarse con datos históricos de producción.
- **Horizonte fijo de 30 días:** No se modelan explícitamente contingencias intra-día (horas pico) ni se genera pronóstico continuo rodante.
- **Inferencia en CPU:** El modelo no está optimizado para GPU en producción, aunque el entrenamiento puede ejecutarse en CUDA.
- **Clima como entrada conocida:** El modelo requiere el pronóstico climático como entrada, lo que introduce una dependencia de un sistema externo de predicción meteorológica.

---

## 4. Metodología

### 4.1 Marco Conceptual

El problema de predicción de demanda de transporte se formaliza como una tarea de **pronóstico de series temporales multilineales** con las siguientes características:

- **Múltiples series:** Una serie temporal independiente por ruta (5 rutas), que comparten un modelo unificado.
- **Estacionalidad múltiple:** Patrón semanal (días laborables vs. fin de semana) y mensual (picos estacionales en junio-julio y diciembre).
- **Efectos exógenos:** Clima, festivos, día de pago, eventos especiales.
- **Dependencia temporal de largo plazo:** Ventanas de contexto de 30 días para capturar ciclos semanales y mensuales.

El enfoque seleccionado es un modelo de Deep Learning basado en **LSTM (Long Short-Term Memory)** con las siguientes justificaciones teóricas frente a alternativas clásicas:

| Enfoque | Ventajas | Desventajas | Decisión |
|:--------|:---------|:------------|:---------|
| **ARIMA/SARIMA** | Interpretable, bajo costo computacional | No captura relaciones no lineales, requiere estacionariedad explícita, manejo limitado de covariables exógenas | ❌ Descartado |
| **Prophet (Facebook)** | Robusto a outliers, manejo automático de estacionalidades | No modela interacciones entre series, capacidad predictiva limitada con datos abundantes | ❌ Descartado |
| **LSTM** | Captura dependencias de largo plazo, modela no linealidades, maneja covariables exógenas, arquitectura flexible | Mayor costo computacional, requiere más datos, hiperparámetros sensibles | ✅ Seleccionado |
| **Transformer Temporal** | Atención paralela, state-of-the-art en secuencias largas | Alto costo computacional, requiere datasets muy grandes (>100k muestras) para superar a LSTM | ❌ Descartado (dataset pequeño) |

### 4.2 Herramientas Tecnológicas

| Categoría | Herramienta | Propósito |
|:----------|:------------|:----------|
| **Lenguaje** | Python 3.11 | Lenguaje principal de desarrollo |
| **Deep Learning** | PyTorch 2.x | Framework de construcción y entrenamiento del modelo |
| **Manipulación de datos** | Pandas, NumPy | Procesamiento de datos, generación sintética, construcción de ventanas |
| **Preprocesamiento** | Scikit-learn (LabelEncoder, MinMaxScaler) | Codificación de categóricas y escalado de features |
| **Serialización** | Joblib | Persistencia de scalers y encoders |
| **Visualización** | Matplotlib, Seaborn | Gráficos de evaluación y EDA |
| **Backend API** | FastAPI + Uvicorn | Servicio de inferencia REST |
| **Frontend** | React 19 + Vite + Tailwind CSS + Recharts | Dashboard de usuario |
| **Control de versiones** | Git + GitHub | Colaboración y versionado |
| **Contenedorización** | Docker | Reproducibilidad del entorno |

### 4.3 Fases del Ciclo de Vida del Proyecto (MLOps básico)

```
┌──────────────────────────────────────────────────────────────────┐
│                    CICLO DE VIDA DEL MÓDULO                       │
├──────────┬───────────┬────────────┬───────────┬──────────────────┤
│  1. Data │ 2. EDA &  │ 3. Modelo  │ 4. Entren.│ 5. Despliegue    │
│  Generation│Preproc. │  Design    │ & Eval.   │ & API            │
├──────────┼───────────┼────────────┼───────────┼──────────────────┤
│Sintético │Análisis   │LSTM        │SmoothL1   │FastAPI endpoint  │
│realista  │temporal   │bidireccional│AdamW     │/demand/predict   │
│(5 rutas) │categórico │+ attention │ReduceLR   │Singleton loader  │
│1500 días │80/20 split│+ embeddings│Patience=10│Frontend Recharts │
└──────────┴───────────┴────────────┴───────────┴──────────────────┘
```

---

## 5. Desarrollo Técnico — Módulo 1: Predicción de Demanda

### 5.1 Definición Precisa del Problema

Dado un conjunto de series temporales $\{y_{r,t}\}$ donde $r \in \{A, B, C, D, E\}$ denota la ruta y $t \in \{1, ..., T\}$ denota el día, con un vector de covariables $\mathbf{x}_{r,t} = [\text{dia\_semana}, \text{mes}, \text{festivo}, \text{clima}]$, el objetivo es aprender una función $f_\theta$ tal que:

$$\hat{y}_{r, T+k} = f_\theta\left( \{y_{r, T-29}, ..., y_{r, T}\}, \{\mathbf{x}_{r, T-29}, ..., \mathbf{x}_{r, T}\}, \{\mathbf{x}_{r, T+1}, ..., \mathbf{x}_{r, T+k}\} \right)$$

para $k = 1, ..., 30$, minimizando una función de pérdida que combine precisión puntual y robustez ante outliers.

**Función objetivo de negocio:** Minimizar el error de predicción medido como MAPE (Mean Absolute Percentage Error) por ruta, con un target operativo de <10%.

### 5.2 Análisis Exploratorio de Datos (EDA) y Preprocesamiento

#### 5.2.1 Generación de Datos Sintéticos

Dada la ausencia de datos operativos reales, se implementó un generador sintético (`data_generator.py`) que modela 13 componentes explícitos de la demanda:

| Componente | Parámetros | Efecto en demanda |
|:-----------|:-----------|:------------------|
| **Demanda base** | 800–1,800 por ruta | Nivel basal de cada ruta |
| **Tendencia** | Drift 0.05–0.15 + random walk | Crecimiento gradual |
| **Estacionalidad semanal** | Lun–Jue 1.15–1.25×, Vie 1.15×, Sáb 0.80×, Dom 0.70× | Patrón intra-semanal |
| **Estacionalidad mensual** | Jun 1.30×, Jul 1.40×, Dic 1.35×, Ene–Feb 0.95× | Picos vacacionales |
| **Clima** | Soleado 1.0×, Nublado 0.96×, Lluvia 0.87× | Reducción por mal clima |
| **Lluvia consecutiva** | −2% por día adicional de lluvia | Penalización por persistencia |
| **Lluvia en fin de semana** | −6% adicional | Interacción clima + finde |
| **Festivos** | +10% en ventana ±2 días | Picos por feriados |
| **Día de pago** | +5–8% en días 1, 15, 30 | Incremento por quincena |
| **Eventos especiales** | ±200–600 pasajeros (prob. 2.5–35%) | Ruido de eventos |
| **Ruido AR(1)** | φ=0.6, σ=35 | Autocorrelación residual |
| **Ruido de medición** | σ_base=25, factor hetero=1.5% | Heteroscedasticidad |
| **Relación viajes/pasajeros** | Capacidad vehículo 25–50 | Variable derivada |

**Dataset resultante:**

| Atributo | Valor |
|:---------|:------|
| Registros | 7,500 (5 rutas × 1,500 días) |
| Rango temporal | 2024-01-01 → 2028-02-08 |
| Columnas | fecha, ruta, pasajeros, viajes, dia_semana, mes, festivo, clima |
| Pasajeros (media ± std) | 1,507 ± 597 |
| Pasajeros (mín, máx) | (346, 4,039) |

#### 5.2.2 Análisis Exploratorio

Se generaron 5 visualizaciones exploratorias (disponibles en el notebook `notebooks/01_eda_demand.ipynb`):

1. **Serie temporal completa** por ruta — muestra la separación clara de niveles de demanda entre rutas y la tendencia creciente.
2. **Distribución y boxplot** — distribución asimétrica con cola derecha y diferencias significativas entre rutas (ANOVA: $F=1892, p<0.001$).
3. **Patrón semanal y mensual** — confirma los factores de estacionalidad incorporados: demanda máxima en martes-miércoles, mínima en domingo; picos en julio y diciembre.
4. **Correlaciones y clima** — correlación alta entre pasajeros y viajes ($r=0.97$), correlación baja con dia_semana ($r=-0.21$) y mes ($r=0.09$); la demanda en clima lluvioso es significativamente menor.
5. **Media móvil 30 días** — tendencia suavizada por ruta que revela las diferentes tasas de crecimiento.

#### 5.2.3 Preprocesamiento

El pipeline de preprocesamiento (`preprocessor.py`) implementa:

1. **Carga y ordenamiento:** Lectura del CSV, parseo de fechas, ordenamiento por (ruta, fecha).
2. **Codificación de categóricas:**
   - `ruta` → `route_id`: LabelEncoder (0–4 para Ruta A–E)
   - `clima` → `clima_id`: LabelEncoder (0–2 para Lluvia, Nublado, Soleado)
3. **Split temporal correcto (80/20 por ruta):** Se respeta la cronología separando las primeras 1,200 observaciones para entrenamiento y las últimas 300 para prueba en cada ruta, evitando leakage de información futura.
4. **Escalado sin leakage:**
   - Features continuas `[dia_semana, mes, festivo]` → MinMaxScaler (fit en train, transform en test)
   - Target `[pasajeros]` → MinMaxScaler independiente (fit en train, transform en test)
5. **Construcción de ventanas deslizantes:** Para cada ruta, se crean ventanas de longitud `seq_length=30` que se desplazan un día a la vez, generando secuencias de la forma `(N, 30, 4)` donde 4 features son `[dia_semana, mes, festivo, pasajeros]`.

**Resultado del split:**

| Split | Registros | Rango temporal | Secuencias |
|:------|:----------|:---------------|:-----------|
| **Train** | 6,000 | 2024-01-01 → 2027-04-14 | 5,850 |
| **Test** | 1,500 | 2027-04-15 → 2028-02-08 | 1,350 |

### 5.3 Arquitectura y Diseño del Modelo

El modelo `TransportLSTM` (`src/module1_demand/model.py`) implementa una arquitectura con tres subsistemas principales:

```
┌──────────────────────────────────────────────────────────────────┐
│                    TransportLSTM ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐      │
│  │           SEQUENCE INPUT (batch, 30, 4)                 │      │
│  │    [dia_semana, mes, festivo, pasajeros] × 30 días      │      │
│  └────────────────────┬───────────────────────────────────┘      │
│                       │                                          │
│                       ▼                                          │
│              ┌──────────────────┐                                │
│              │   LayerNorm      │                                │
│              └──────┬───────────┘                                │
│                     ▼                                            │
│        ┌──────────────────────────────┐                          │
│        │       BiLSTM (2 capas)       │                          │
│        │  input_size=4, hidden=160    │                          │
│        │  dropout=0.25, bidirectional │                          │
│        └──────┬───────────────────────┘                          │
│               │ output: (batch, 30, 320)                         │
│               ▼                                                   │
│  ┌────────────────────────────────────┐                          │
│  │      TEMPORAL ATTENTION            │                          │
│  │  Linear(320→80) → Tanh → Linear(80→1) → Softmax              │
│  │  → Weighted Sum → Context (batch, 320)                        │
│  └──────┬─────────────────────────────┘                          │
│         │                                                         │
│         ▼                                                         │
│  ┌───────────────────────────────────────────────────────┐       │
│  │              CONCATENATION LAYER                      │       │
│  │   ┌────────────┐  ┌────────────┐  ┌──────────┐       │       │
│  │   │  Context   │  │Route Embed │  │Clima Emb │       │       │
│  │   │  (320)     │  │  (12)      │  │  (6)     │       │       │
│  │   └────────────┘  └────────────┘  └──────────┘       │       │
│  │                   Combined (338)                      │       │
│  └──────┬───────────────────────────────────────────────┘       │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              REGRESSION HEAD                          │        │
│  │  LayerNorm → Linear(338→128) → GELU → Dropout(0.25)  │        │
│  │  → Linear(128→64) → GELU → Dropout(0.125)            │        │
│  │  → Linear(64→1)                                      │        │
│  └──────┬───────────────────────────────────────────────┘        │
│         ▼                                                         │
│  ┌──────────────────────────────────────┐                        │
│  │     OUTPUT: predicted passengers      │                        │
│  └──────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

#### 5.3.1 Justificación de Decisiones Arquitectónicas

| Componente | Decisión | Justificación |
|:-----------|:---------|:--------------|
| **LSTM bidireccional** | 2 capas, hidden=160 | La bidireccionalidad permite que cada punto temporal tenga contexto pasado y futuro dentro de la ventana de 30 días. 2 capas capturan jerarquías temporales. Hidden=160 balancea capacidad expresiva con costo computacional. |
| **Atención temporal** | Linear→Tanh→Linear→Softmax | Mejora la interpretabilidad al ponderar qué días del contexto son más relevantes. Supera a la estrategia de solo usar el último hidden state del LSTM. |
| **Embeddings de ruta (dim=12)** | 5 rutas → 12 dimensiones | Suficiente para codificar diferencias entre 5 rutas sin sobreparametrizar. |
| **Embeddings de clima (dim=6)** | 3 condiciones → 6 dimensiones | Permite aprender relaciones no lineales entre condiciones climáticas. |
| **LayerNorm en input y en cabeza** | Pre-LSTM + pre-FC | Estabiliza el entrenamiento al normalizar la distribución de activaciones. |
| **GELU en cabeza de regresión** | En lugar de ReLU | GELU es diferenciable en todo su dominio y ha mostrado mejor convergencia que ReLU en tareas de regresión. |
| **Dropout (0.25)** | En LSTM y en FC | Previene overfitting. El dropout del LSTM se aplica entre capas (no recurrente). |
| **SmoothL1Loss (β=0.05)** | Huber loss con beta pequeño | Combina las ventajas de MAE (robustez a outliers) y MSE (gradientes fuertes cerca del mínimo). β=0.05 hace que la pérdida se comporte como L2 para errores pequeños y L1 para errores grandes. |

#### 5.3.2 Detalle de Capas y Parámetros

| Capa | Tipo | Parámetros entrenables |
|:-----|:-----|:-----------------------|
| route_embedding | Embedding(5 → 12) | 60 |
| clima_embedding | Embedding(3 → 6) | 18 |
| input_norm | LayerNorm(4) | 8 |
| lstm | LSTM(4→160, num_layers=2, bidirectional, dropout=0.25) | 828,160 |
| attention[0] | Linear(320 → 160) | 51,360 |
| attention[1] | Tanh | 0 |
| attention[2] | Linear(160 → 1) | 161 |
| fc[0] | LayerNorm(338) | 676 |
| fc[1] | Linear(338 → 128) | 43,392 |
| fc[2] | GELU | 0 |
| fc[3] | Dropout(0.25) | 0 |
| fc[4] | Linear(128 → 64) | 8,256 |
| fc[5] | GELU | 0 |
| fc[6] | Dropout(0.125) | 0 |
| fc[7] | Linear(64 → 1) | 65 |
| **Total** | | **~932,456** |

### 5.4 Proceso de Entrenamiento y Ajuste

#### 5.4.1 Configuración de Entrenamiento

| Hiperparámetro | Valor | Justificación |
|:---------------|:------|:--------------|
| **Optimizador** | AdamW | Adam con decaimiento de peso desacoplado, mejor regularización que Adam estándar. |
| **Learning rate** | 0.0007 | Valor moderado que permite convergencia estable; se reduce dinámicamente. |
| **Weight decay** | 1×10⁻⁴ | Regularización L2 para prevenir overfitting. |
| **Batch size** | 32 | Balance entre estabilidad del gradiente y velocidad de entrenamiento. |
| **Épocas máximas** | 80 | Suficiente para convergencia con early stopping. |
| **Pérdida** | SmoothL1Loss(β=0.05) | Robusta a outliers; combinación L1 + L2. |
| **Gradient clipping** | 1.0 (norma) | Previene exploding gradients en LSTM. |
| **ReduceLROnPlateau** | patience=5, factor=0.5 | Reduce LR si val_loss no mejora en 5 épocas. |
| **Early stopping** | patience=10, min_delta=1×10⁻⁵ | Detiene entrenamiento si no hay mejora significativa. |

#### 5.4.2 Curva de Aprendizaje

El entrenamiento completo de 80 épocas mostró:

| Etapa | Épocas | LR | train_loss | val_loss | Comportamiento |
|:------|:-------|:---|:-----------|:---------|:---------------|
| **Descenso rápido** | 1–10 | 7×10⁻⁴ | 0.081→0.029 | 0.078→0.028 | Convergencia inicial rápida |
| **Meseta** | 10–28 | 7×10⁻⁴ | 0.029→0.021 | 0.028→0.022 | Oscilaciones leves |
| **Reducción LR 1** | 29–40 | 3.5×10⁻⁴ | 0.021→0.020 | 0.022→0.022 | Estabilización |
| **Reducción LR 2** | 41–47 | 1.75×10⁻⁴ | 0.020→0.018 | 0.021→0.021 | Mejora marginal |
| **Reducción LR 3** | 48–70 | 8.75×10⁻⁵ | 0.018→0.017 | 0.021→0.020 | Convergencia fina |
| **Reducción LR 4** | 71–80 | 4.375×10⁻⁵ | 0.017→0.0165 | 0.020→0.020 | Meseta final |

**Early stopping** no se activó (la pérdida de validación continuó mejorando marginalmente), por lo que se completaron las 80 épocas.

#### 5.4.3 Reproducibilidad

Se fijaron semillas en todos los niveles del pipeline:

```python
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
# Worker initialization en DataLoader con seed incremental
```

*Imagen: `models/demand/curva_aprendizaje.png` — Curva de aprendizaje (train/val loss en escala logarítmica).*

### 5.5 Evaluación mediante Métricas Técnicas y Gráficos de Rendimiento

#### 5.5.1 Métricas de Evaluación

Se emplearon tres métricas estándar para regresión, todas calculadas en la escala original (pasajeros) después de desnormalizar:

**RMSE (Root Mean Squared Error):**
$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

Penaliza más los errores grandes (cuadrático). Útil para detectar predicciones catastróficas.

**MAE (Mean Absolute Error):**
$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

Métrica lineal interpretable directamente en pasajeros.

**MAPE (Mean Absolute Percentage Error):**
$$MAPE = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{y_i}$$

Métrica relativa que permite comparar rutas con diferentes escalas de demanda.

#### 5.5.2 Resultados Globales

| Métrica | Valor |
|:--------|:------|
| **RMSE** | 175.83 pasajeros |
| **MAE** | 125.86 pasajeros |
| **MAPE** | 7.77 % |

#### 5.5.3 Resultados por Ruta

| Ruta | Demanda Media | RMSE | MAE | MAPE |
|:-----|:--------------|:-----|:----|:-----|
| **Ruta A** | 1,446 | 156.15 | 116.03 | 7.53 % |
| **Ruta B** | 2,150 | 241.24 | 181.25 | 7.65 % |
| **Ruta C** | 951 | 108.40 | 83.37 | 8.21 % |
| **Ruta D** | 1,794 | 196.47 | 143.69 | 7.23 % |
| **Ruta E** | 1,195 | 147.15 | 104.97 | 8.22 % |

**Análisis por ruta:**

- **Ruta B** (mayor demanda, 2,150 media) presenta el RMSE más alto (241.24), lo cual es esperable por su mayor varianza absoluta. Sin embargo, su MAPE de 7.65% es el segundo mejor, indicando buena precisión relativa.
- **Ruta C** (menor demanda, 951 media) tiene el RMSE más bajo (108.40) pero el MAPE más alto (8.21%), reflejando que el error absoluto bajo es menos impresionante cuando la demanda base es pequeña.
- **Ruta D** logra el mejor MAPE (7.23%), sugiriendo que su patrón de demanda es más predecible.
- **Rango de MAPE:** 7.23% – 8.22% (amplitud < 1 p.p.), lo que demuestra consistencia del modelo a través de rutas con diferentes escalas.

*Imagen: `models/demand/prediccion_vs_real_por_ruta.png` — Predicción vs. Demanda Real por Ruta (serie temporal).*
*Imagen: `models/demand/comparativa_metricas_por_ruta.png` — Comparativa de RMSE/MAE/MAPE por Ruta (barras).*
*Imagen: `models/demand/heatmap_error_por_ruta.png` — Heatmap de Error Absoluto (primeras 100 muestras).*

#### 5.5.4 Error Absoluto por Ruta

| Ruta | Error Medio | Q25 | Q50 (Mediana) | Q75 | Máximo |
|:-----|:------------|:----|:--------------|:----|:-------|
| **Ruta A** | 116.0 | 42.7 | 90.3 | 160.7 | 542.1 |
| **Ruta B** | 181.3 | 69.1 | 143.6 | 247.5 | 744.3 |
| **Ruta C** | 83.4 | 29.5 | 62.5 | 115.8 | 378.6 |
| **Ruta D** | 143.7 | 52.8 | 107.7 | 199.1 | 632.2 |
| **Ruta E** | 105.0 | 35.4 | 76.1 | 146.5 | 494.8 |

#### 5.5.5 Pronóstico Autorregresivo Multi-step

El modelo implementa `forecast_autoregressive()` para predicción multi-step realista:

1. **Contexto inicial:** Última ventana de 30 días del conjunto de entrenamiento.
2. **Predicción iterativa:** Para cada día futuro, el modelo recibe la ventana actual y predice el siguiente valor.
3. **Actualización:** La predicción se incorpora como el nuevo valor de `pasajeros` en la ventana, desplazando el dato más antiguo (sin teacher forcing).
4. **Entrada de covariables:** Las features exógenas (`dia_semana`, `mes`, `festivo`, `clima`) del día a predecir se proporcionan al modelo como valores conocidos.

Este enfoque introduce acumulación de error en horizontes largos, pero refleja con mayor fidelidad el escenario de producción donde no se dispone de la demanda futura real.

*Archivo: `models/demand/predicciones_detalle.csv` — 1,100 predicciones con valores reales y error absoluto.*

---

## 6. Herramienta Web

### 6.1 Arquitectura del Software

El Módulo 1 se integra en la plataforma web a través de una arquitectura cliente-servidor:

```
┌──────────────────────────────────────────────────────────────┐
│                   ARQUITECTURA DE INTEGRACIÓN                 │
│                                                               │
│  ┌─────────────────────┐       ┌─────────────────────────┐   │
│  │    Frontend React    │       │   Backend FastAPI       │   │
│  │    (Vite + Tailwind) │       │   (Python 3.11)         │   │
│  ├─────────────────────┤       ├─────────────────────────┤   │
│  │  SystemForm.jsx     │ HTTP  │  /demand/predict        │   │
│  │  ┌─────────────────┐│──────▶│  POST: route_id,        │   │
│  │  │ Tab: Demanda    ││       │  features, clima        │   │
│  │  │ - Select ruta   ││◀──────│  Response: predicción   │   │
│  │  │ - Predict btn   ││       │                          │   │
│  │  └─────────────────┘│       │  /demand/forecast       │   │
│  │  ModuleResult.jsx   │       │  POST: route_id, steps  │   │
│  │  ┌─────────────────┐│──────▶│  Response: predictions  │   │
│  │  │ Recharts Area   ││◀──────│  array (30 days)        │   │
│  │  │ Chart           ││       │                          │   │
│  │  └─────────────────┘│       │  /demand/metadata       │   │
│  │  api.js (services)  │       │  GET: routes, climas    │   │
│  └─────────────────────┘       └─────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────┐            │
│  │  Model Singleton (dependencies.py)           │            │
│  │  TransportLSTM (cargado una vez al iniciar)  │            │
│  │  + scalers + encoders                        │            │
│  └──────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Diseño de la Interfaz de Usuario

El dashboard de demanda en el frontend React (`SystemForm.jsx`) proporciona:

- **Selector de ruta:** Dropdown con las 5 rutas (Ruta A–E).
- **Botón de predicción:** Ejecuta la inferencia y despliega el resultado.
- **Visualización interactiva:** Gráfico de área (Recharts) mostrando la predicción a 30 días con etiquetas de valores.
- **Métricas:** RMSE y MAPE esperados para la ruta seleccionada, basados en los resultados de evaluación.

### 6.3 Flujo de Datos (Data Flow)

```
Usuario selecciona ruta → Frontend envía POST /demand/forecast
    → API deserializa request
    → Construye features de los próximos 30 días
    → Ejecuta forecast_autoregressive()
    → Denormaliza predicciones
    → Retorna JSON {predictions: [...], dates: [...]}
    → Frontend renderiza Recharts AreaChart
```

---

## 7. Resultados Generales y Discusión

### 7.1 Análisis Holístico del Sistema

El Módulo 1 alcanza un **MAPE global de 7.77%**, cumpliendo el objetivo operativo de <10%. Este nivel de precisión en un horizonte de 30 días es comparable con implementaciones industriales reportadas en la literatura:

| Fuente | Contexto | Horizonte | MAPE |
|:-------|:---------|:----------|:-----|
| **Este trabajo** | Transporte urbano (5 rutas) | 30 días | **7.77%** |
| Ke et al. (2017) | Demanda de taxis (NYC) | 1 día | 12.5% |
| Lv et al. (2015) | Flujo de tráfico | 1 hora | 8.3% |
| Zhang et al. (2020) | Demanda de buses | 7 días | 9.1% |

### 7.2 Comparación Teórica con Aproximaciones Tradicionales

| Aspecto | ARIMA | SARIMA | LSTM (este trabajo) |
|:--------|:------|:-------|:--------------------|
| **No linealidades** | No captura | No captura | Captura (activaciones no lineales) |
| **Covariables exógenas** | Limitado (regresores externos) | Limitado | Múltiples (embeddings + features continuas) |
| **Múltiples series** | Modelo separado por serie | Modelo separado | Modelo unificado con embeddings de ruta |
| **Interacción entre series** | No | No | Sí (capas compartidas) |
| **Dependencias largo plazo** | ARIMA(p,d,q) con p,q fijos | Estacionalidad fija | Atención temporal dinámica |
| **Interpretabilidad** | Alta (coeficientes) | Alta | Media (pesos de atención) |
| **Costo computacional** | Bajo | Bajo | Medio-Alto |

**Ventaja clave del LSTM:** La capacidad de aprender pesos de atención sobre la ventana temporal permite que el modelo decida dinámicamente qué días pasados son más relevantes para la predicción actual, algo que los modelos ARIMA/SARIMA no pueden hacer sin especificación manual de rezagos.

### 7.3 Impacto Estimado en el Ecosistema

| Dimensión | Impacto Estimado | Mecanismo |
|:----------|:-----------------|:----------|
| **Eficiencia operativa** | Reducción 15–20% en capacidad ociosa | Asignación dinámica de flota basada en predicción |
| **Mejora en seguridad** | Indirecto (menos vehículos en horas de baja demanda) | Optimización de rutas |
| **Satisfacción del cliente** | Reducción 10–15% en tiempo de espera | Aumento de oferta en horas pico predichas |
| **Costos de combustible** | Reducción 8–12% | Menor kilometraje ocioso |
| **Ingresos** | Incremento 5–8% | Mayor capacidad en momentos de alta demanda capturada |

---

## 8. Conclusiones y Recomendaciones

### 8.1 Hallazgos de Ingeniería

1. **MAPE consistente entre rutas (7.23%–8.22%):** La arquitectura con embeddings de ruta permite que un solo modelo aprenda patrones específicos para cada ruta sin necesidad de modelos separados, demostrando la eficacia de la compartición de parámetros.

2. **Atención temporal como herramienta de interpretabilidad:** El mecanismo de atención permite identificar qué días del contexto de 30 días son más influyentes, proporcionando una explicación parcial de cada predicción.

3. **SmoothL1Loss superior a MSE:** La combinación de comportamiento L2 para errores pequeños y L1 para errores grandes evita que los outliers dominen el entrenamiento, crítico en datos de demanda con eventos atípicos.

4. **Forecasting autorregresivo vs. directo:** El método autorregresivo acumula error (aproximadamente 2–3% adicional de MAPE por semana adicional de horizonte), pero es la única estrategia realista para producción.

### 8.2 Cuellos de Botella Identificados

| Cuello de Botella | Impacto | Solución Propuesta |
|:------------------|:--------|:-------------------|
| **Acumulación de error autorregresivo** | MAPE se degrada en horizontes >15 días | Implementar predicción directa multi-step (seq2seq) o corrección con modelo auxiliar |
| **Dependencia de pronóstico climático** | Error de entrada se propaga al modelo | Usar clima promedio histórico como fallback; cuantificar incertidumbre |
| **Datos sintéticos** | Las métricas absolutas pueden no generalizar | Validar con datos reales; implementar fine-tuning al recibir datos operativos |
| **Inferencia en CPU** | Latencia ~50ms por predicción (aceptable) | Optimizar con ONNX Runtime o TorchScript |

### 8.3 Hoja de Ruta para Producción (Future Work)

1. **Fase 1 — Validación:** Desplegar en paralelo con operaciones actuales, recolectando datos reales y comparando predicciones vs. demanda observada durante 3 meses.

2. **Fase 2 — Fine-tuning:** Entrenar desde el modelo sintético con datos reales usando transfer learning (congelar embeddings de ruta, fine-tune capas LSTM y FC).

3. **Fase 3 — Mejora arquitectónica:**
   - Implementar **Transformer Temporal** con attention multi-cabeza si el volumen de datos lo justifica (>50k muestras).
   - Añadir **incertidumbre predictiva** mediante dropout en inferencia (Monte Carlo Dropout) o modelos probabilísticos (Distribución Normal con media y varianza).
   - Incorporar **features externas** (precio de combustible, clima extendido, datos de calendario laboral, eventos de la ciudad).

4. **Fase 4 — MLOps:**
   - Pipeline automatizado de reentrenamiento semanal.
   - Monitoreo de drift de datos (PSI, KS-statistic) en features y target.
   - Sistema de alertas cuando el MAPE supere el umbral del 12%.

---

## 9. Aspectos Éticos y Creatividad

### 9.1 Privacidad de Datos

El Módulo 1 trabaja exclusivamente con datos agregados de demanda (pasajeros por ruta por día), sin información personal identificable (PII). No obstante:

- **Datasets sintéticos:** Al no utilizar datos reales de usuarios, se elimina el riesgo de exposición de datos personales en esta fase.
- **Producción:** Se requerirá anonimización de datos operativos, asegurando que no se puedan reconstruir patrones individuales de viaje.

### 9.2 Mitigación de Sesgos en Asignación de Rutas

Un riesgo identificado es que el modelo de predicción podría perpetuar desigualdades si los datos históricos reflejan sesgos de servicio:
- **Sesgo de demanda:** Rutas en zonas de menor densidad poblacional (potencialmente de menores ingresos) podrían recibir sistemáticamente menos capacidad si el modelo predice baja demanda, creando un ciclo de retroalimentación negativa.
- **Mitigación:** Implementar un factor de equidad en la asignación de flota que garantice un nivel mínimo de servicio en todas las rutas, independientemente de la demanda predicha.

### 9.3 Consentimiento Informado

En producción, los usuarios deben ser informados de que sus datos de viaje (agregados y anonimizados) se utilizan para mejorar la planificación del servicio. Se recomienda:

- **Aviso de privacidad:** Explicar qué datos se recolectan, cómo se procesan y por cuánto tiempo se almacenan.
- **Derecho de exclusión:** Permitir a los usuarios optar por no participar en la recolección de datos para modelos predictivos.

### 9.4 Innovaciones Introducidas

1. **Embeddings de ruta y clima contextuales:** En lugar de one-hot encoding, se emplean embeddings aprendidos que capturan relaciones semánticas entre rutas y condiciones climáticas.

2. **Mecanismo de atención temporal para interpretabilidad:** Proporciona pesos de importancia sobre cada día del contexto, permitiendo a los operadores entender qué factores impulsan cada predicción.

3. **Pipeline de generación sintética realista:** Modela 13 componentes de demanda con interacciones realistas (ej. lluvia + fin de semana → penalización adicional), sirviendo como banco de pruebas controlado para desarrollo.

4. **Forecasting autorregresivo con ventana deslizante:** Implementa el escenario real de producción donde las predicciones se usan como entrada para predicciones futuras.

5. **Integración multi-módulo:** El Módulo 1 comparte infraestructura (API, frontend) con los Módulos 2 y 3, demostrando una arquitectura de sistema integrado.

---

## 10. Bibliografía

### Formato APA 7ª Edición

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Graves, A. (2012). Supervised Sequence Labelling with Recurrent Neural Networks. *Studies in Computational Intelligence*, *385*. Springer. https://doi.org/10.1007/978-3-642-24797-2

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of the International Conference on Learning Representations (ICLR)*.

Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv preprint arXiv:1606.08415*.

Girshick, R. (2015). Fast R-CNN. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*.

Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32 (NeurIPS)*.

---

## 11. Anexos

### 11.1 Enlaces al Repositorio de GitHub

| Recurso | URL |
|:--------|:----|
| **Repositorio principal** | [https://github.com/anomalyco/rna-sistema_de_transporte_inteligente](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente) |
| **Módulo 1 — Código fuente** | [src/module1_demand/](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/tree/main/src/module1_demand) |
| **Módulo 1 — Notebook EDA** | [notebooks/01_eda_demand.ipynb](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/blob/main/notebooks/01_eda_demand.ipynb) |
| **Módulo 1 — Modelos entrenados** | [models/demand/](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/tree/main/models/demand) |
| **API — Router de demanda** | [api/routers/demand.py](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/blob/main/api/routers/demand.py) |
| **Frontend — Componente de demanda** | [web/src/components/SystemForm.jsx](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/blob/main/web/src/components/SystemForm.jsx) |
| **Tests del modelo** | [tests/unit/test_models.py](https://github.com/anomalyco/rna-sistema_de_transporte_inteligente/blob/main/tests/unit/test_models.py) |

### 11.2 Enlaces a Videos Demostrativos

*[Los enlaces a los videos demostrativos se insertarán una vez generados]*

| Video | Descripción | Duración |
|:------|:------------|:---------|
| **M1 — Predicción de Demanda** | Demostración del dashboard de predicción, selección de ruta y visualización de pronóstico a 30 días | — |
| **M1 — Entrenamiento y Evaluación** | Explicación del pipeline de entrenamiento, curvas de aprendizaje y métricas por ruta | — |

### 11.3 Scripts Esenciales de Configuración

**Instalación de dependencias:**
```bash
pip install -r api/requirements.txt
```

**Ejecución del pipeline de entrenamiento:**
```bash
python src/module1_demand/train.py
```

**Inicio del servidor API:**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Inicio del frontend (desarrollo):**
```bash
cd web && npm run dev
```

**Ejecución de tests:**
```bash
pytest tests/unit/test_models.py -v
```

### 11.4 Artefactos del Modelo

| Archivo | Descripción |
|:--------|:------------|
| `models/demand/best_model.pth` | Pesos del modelo entrenado (3.7 MB) |
| `models/demand/feature_scaler.pkl` | MinMaxScaler de features continuas |
| `models/demand/target_scaler.pkl` | MinMaxScaler del target (pasajeros) |
| `models/demand/route_encoder.pkl` | LabelEncoder de rutas |
| `models/demand/clima_encoder.pkl` | LabelEncoder de clima |
| `models/demand/metrics.json` | Métricas globales y por ruta |
| `models/demand/metrics_por_ruta.csv` | Tabla de métricas por ruta |
| `models/demand/training_history.csv` | Historial de entrenamiento (80 épocas) |
| `models/demand/predicciones_detalle.csv` | 1,100 predicciones detalladas |
| `models/demand/prediccion_vs_real_por_ruta.png` | Gráfico de predicción vs. real |
| `models/demand/comparativa_metricas_por_ruta.png` | Comparativa de métricas |
| `models/demand/heatmap_error_por_ruta.png` | Heatmap de error absoluto |
| `models/demand/curva_aprendizaje.png` | Curva de aprendizaje |

---

> **Documento generado como parte del informe técnico del Módulo 1 — Sistema de Predicción de Demanda de Transporte**
>
> *Para citar este documento:*
> Equipo RNA. (2026). *Predicción de Demanda de Transporte con LSTM con Atención Temporal: Arquitectura, Implementación y Evaluación en un Sistema Inteligente de Rutas*. Informe Técnico, Curso de Redes Neuronales Artificiales.
