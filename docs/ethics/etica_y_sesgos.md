# Reflexión Ética sobre el Manejo de Datos y Sesgos Algorítmicos

## Sistema Inteligente de Transporte

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Institución:** Universidad Nacional de Colombia  
**Fecha:** Mayo de 2026

---

## 1. Introducción

El desarrollo de sistemas de inteligencia artificial para transporte implica el procesamiento de datos de pasajeros, imágenes de conductores e historiales de viaje de usuarios. Esta reflexión analiza los riesgos éticos asociados con cada módulo del sistema y propone medidas de mitigación concretas.

---

## 2. Privacidad y Protección de Datos

### 2.1 Módulo de demanda

El módulo de predicción de demanda opera sobre datos agregados de pasajeros por ruta y fecha. Aunque el dataset actual es sintético, en producción estos datos provendrían de sistemas de recaudo, GPS vehicular y conteo de ocupación. Los riesgos incluyen:

- **Reidentificación:** datos agregados de rutas pequeñas o horarios específicos podrían permitir inferir patrones de movilidad de comunidades vulnerables o individuos.
- **Vigilancia masiva:** la combinación de datos de demanda con cámaras, GPS y registros de pago puede construir perfiles detallados de movilidad ciudadana.

**Medidas propuestas:**

- Aplicar técnicas de privacidad diferencial al agregar datos de demanda.
- Establecer umbrales mínimos de agregación: no reportar predicciones para rutas o franjas horarias con menos de N usuarios.
- Limitar la retención de datos históricos a un periodo justificado (por ejemplo, 24 meses).
- Cumplir con la Ley 1581 de 2012 de Protección de Datos Personales de Colombia.

### 2.2 Módulo de conducción distractiva

Este módulo procesa imágenes de conductores dentro de cabinas, lo que constituye datos biométricos sensibles. Los riesgos son significativos:

- **Consentimiento:** los conductores pueden no haber dado consentimiento informado para que sus imágenes sean analizadas por un sistema de IA.
- **Vigilancia laboral:** el sistema podría usarse para monitoreo laboral excesivo, afectando la dignidad del trabajador y generando presión psicológica.
- **Retención indebida:** las imágenes y clasificaciones podrían almacenarse indefinidamente y usarse para decisiones disciplinarias sin contexto.

**Medidas propuestas:**

- Implementar consentimiento informado explícito antes de capturar imágenes.
- Procesar imágenes en tiempo real sin almacenamiento permanente cuando sea posible.
- Anonimizar rostros antes del análisis si la detección no requiere identidad.
- Establecer que las clasificaciones automáticas sean insumos para revisión humana, nunca la única base para sanciones.
- Definir políticas claras de acceso, retención y eliminación de registros.

### 2.3 Módulo de recomendación

El recomendador procesa historiales de viaje, calificaciones y preferencias de usuarios. Los riesgos incluyen:

- **Perfilamiento comercial:** los perfiles de viaje pueden revelar información sensible como nivel socioeconómico, religión, salud o relaciones personales.
- **Venta o cesión de datos:** sin controles, los perfiles podrían compartirse con terceros para publicidad o segmentación no autorizada.
- **Manipulación:** recomendaciones dirigidas podrían influir en decisiones de viaje de forma no transparente, priorizando intereses comerciales sobre el bienestar del usuario.

**Medidas propuestas:**

- Informar claramente qué datos se recopilan, con qué fin y por cuánto tiempo se conservan.
- Permitir al usuario acceder, corregir y eliminar sus datos personales.
- No compartir perfiles con terceros sin consentimiento explícito.
- Incluir en cada recomendación una explicación breve de por qué se sugiere ese destino.

---

## 3. Sesgos Algorítmicos y Justicia

### 3.1 Sesgo en el módulo de demanda

- **Rutas subatendidas:** si el modelo se entrena con datos históricos de una empresa que ha invertido menos en ciertas rutas (por ejemplo, rutas a barrios periféricos o zonas rurales), las predicciones reflejarán y perpetuarán esa baja demanda artificial. El modelo podría recomendar reducir frecuencia en rutas que necesitan más inversión.
- **Sesgo geográfico:** las rutas con mayor infraestructura tecnológica (sensores, conteo electrónico) tendrán datos más precisos, mientras que rutas con conteo manual tendrán datos más ruidosos, afectando la calidad de las predicciones.

**Mitigación:**

- Auditar las predicciones por zona geográfica y nivel socioeconómico de las áreas servidas.
- Incorporar variables de equidad: garantizar un nivel mínimo de servicio independientemente de la demanda predicha.
- Validar el modelo con datos de múltiples periodos y contextos para detectar degradación en grupos específicos.

### 3.2 Sesgo en el módulo de conducción distractiva

- **Sesgo demográfico:** el dataset de entrenamiento puede no representar adecuadamente la diversidad de conductores (género, edad, tono de piel, complexión física, uso de gafas, vestimenta cultural). Esto puede causar mayor tasa de falsos positivos o falsos negativos en grupos subrepresentados.
- **Sesgo de cámara:** las imágenes provienen de cámaras con diferentes ángulos, resoluciones e iluminación. El modelo podría funcionar peor en vehículos antiguos con cámaras de menor calidad.
- **Sesgo de contexto:** comportamientos culturalmente específicos (como comer durante la conducción en ciertos contextos laborales) podrían clasificarse de forma inconsistente.

**Mitigación:**

- Evaluar métricas de equidad (equalized odds, demographic parity) por grupo demográfico.
- Ampliar el dataset con imágenes diversas que representen la población real de conductores.
- Implementar umbrales de confianza: clasificaciones con baja confianza deben marcarse para revisión humana.
- Realizar auditorías periódicas de rendimiento por subgrupo.

### 3.3 Sesgo en el módulo de recomendación

- **Sesgo de popularidad:** el recomendador tiende a favorecer destinos turísticos populares (como Taj Mahal, Goa Beaches o Kerala Backwaters), reduciendo la visibilidad de destinos emergentes o menos conocidos. Esto concentra el turismo y puede dañar ecosistemas frágiles por sobreturismo.
- **Sesgo socioeconómico:** si el modelo se entrena con datos de usuarios de cierto perfil económico, las recomendaciones podrían no ser relevantes para usuarios de otros perfiles.
- **Filtro burbuja:** usuarios que han visitado destinos similares recibirán siempre recomendaciones similares, limitando el descubrimiento y la diversidad cultural.

**Mitigación:**

- Medir y optimizar métricas de diversidad intra-lista, cobertura de catálogo y novedad.
- Introducir mecanismos de exploración (epsilon-greedy, Thompson sampling) para balancear explotación y descubrimiento.
- Permitir al usuario ajustar parámetros de personalización vs. exploración.
- Auditar que las recomendaciones no discriminen por perfil demográfico del usuario.

---

## 4. Uso Responsable y Gobernanza

### 4.1 Transparencia

- Documentar públicamente las capacidades y limitaciones de cada módulo.
- Informar a usuarios y conductores cuando una decisión involucre un sistema automático.
- Publicar métricas de rendimiento desglosadas por subgrupo cuando sea aplicable.

### 4.2 Responsabilidad humana

- Las predicciones de demanda deben apoyar decisiones humanas de planeación, no automatizarlas sin supervisión.
- Las clasificaciones de conducción distractiva nunca deben ser la única evidencia para sanciones disciplinarias o legales.
- Las recomendaciones de destinos deben presentarse como sugerencias, no como decisiones vinculantes.

### 4.3 Monitoreo continuo

- Implementar monitoreo de deriva de datos (data drift) para detectar cuándo los datos de producción se desvían del entrenamiento.
- Establecer comités de ética internos que revisen periódicamente el impacto del sistema.
- Crear canales para que usuarios y trabajadores reporten preocupaciones sobre el sistema.

---

## 5. Marco Legal Aplicable

El sistema debe cumplir con:

- **Ley 1581 de 2012** (Colombia): régimen general de protección de datos personales.
- **Decreto 1377 de 2013**: reglamenta el tratamiento de datos personales.
- **Ley 1266 de 2008** (Habeas Data): regula el manejo de información en centrales de riesgo.
- **Resolución 050 de 2024** (MinTIC): lineamientos para el uso ético de IA en el sector público colombiano.
- **Recomendaciones de la OCDE sobre IA** (2019): principios de transparencia, robustez y rendición de cuentas.

---

## 6. Conclusiones

El desarrollo de sistemas de IA para transporte conlleva responsabilidades éticas significativas que van más allá del rendimiento técnico. Los tres módulos del sistema presentan riesgos concretos de sesgo, vigilancia y falta de transparencia que deben abordarse desde la fase de diseño. La implementación de las medidas propuestas —privacidad diferencial, auditorías de equidad, consentimiento informado, revisión humana y monitoreo continuo— es esencial para que el sistema sea técnicamente competente y socialmente responsable.

El equipo de desarrollo reconoce que la ética en IA no es un componente adicional sino un requisito transversal que debe integrarse en cada decisión de diseño, entrenamiento y despliegue.
