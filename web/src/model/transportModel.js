import {
  fetchDemandForecast,
  fetchDriverClassification,
  fetchRecommendations,
} from '../services/api';

export const ROUTES = [
  { id: 0, name: 'Bogotá - Medellín' },
  { id: 1, name: 'Bogotá - Cali' },
  { id: 2, name: 'Bogotá - Cartagena' },
  { id: 3, name: 'Medellín - Cartagena' },
  { id: 4, name: 'Cali - Barranquilla' },
];

export async function predictDemand(routeId) {
  const apiResult = await fetchDemandForecast(routeId, 30);
  const historyRows = apiResult.historico ?? [];
  const historicalPredRows = apiResult.prediccion_historica ?? [];
  const routeName = apiResult.ruta ?? ROUTES.find((r) => r.id === Number(routeId))?.name ?? 'Bogotá - Medellín';

  const days = historyRows.map((r, i) => {
    const date = new Date(r.fecha);
    return {
      day: i + 1,
      date: date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' }),
      real: r.pasajeros,
    };
  });

  const overlapStart = days.length - historicalPredRows.length;
  historicalPredRows.forEach((r, i) => {
    const idx = overlapStart + i;
    if (days[idx]) {
      days[idx].predicted = Math.round(r.prediccion);
    }
  });

  return { days, rmse: null, mae: null, routeName };
}

export const DRIVER_CLASSES = [
  { id: 'safe_driving', label: 'Conducción Segura', icon: '✅', color: '#b8bb26' },
  { id: 'talking_phone', label: 'Hablando por Teléfono', icon: '📱', color: '#fb4934' },
  { id: 'texting_phone', label: 'Escribiendo en Teléfono', icon: '💬', color: '#fe8019' },
  { id: 'turning', label: 'Mirando a los Lados', icon: '👀', color: '#d3869b' },
  { id: 'other_activities', label: 'Otras Actividades', icon: '⚠️', color: '#fabd2f' },
];

export async function classifyDriver(file) {
  const apiResult = await fetchDriverClassification(file);
  const driverClass = DRIVER_CLASSES.find((dc) => dc.id === apiResult.predicted_label) ?? {
    id: apiResult.predicted_label,
    label: apiResult.predicted_label,
    icon: '⚠️',
    color: '#fabd2f',
  };

  return {
    ...driverClass,
    confidence: apiResult.confidence,
    filename: apiResult.filename,
    probabilities: apiResult.probabilities,
    preventiveMeasure: apiResult.preventive_measure,
    predictedLabel: apiResult.predicted_label,
  };
}

export const PREFERENCE_OPTIONS = {
  tripType: [
    { value: 'playa', label: 'Playa y costa' },
    { value: 'montaña', label: 'Montaña y senderismo' },
    { value: 'ciudad', label: 'Ciudad y cultura' },
    { value: 'aventura', label: 'Aventura y ecoturismo' },
    { value: 'espiritual', label: 'Espiritual y patrimonio' },
  ],
  budget: [
    { value: 'bajo', label: 'Económico' },
    { value: 'medio', label: 'Moderado' },
    { value: 'alto', label: 'Premium' },
  ],
  duration: [
    { value: 'corto', label: 'Fin de semana (2-3 días)' },
    { value: 'medio', label: 'Semana (5-7 días)' },
    { value: 'largo', label: 'Quincena o más' },
  ],
  interests: [
    { value: 'gastronomía', label: 'Gastronomía' },
    { value: 'historia', label: 'Historia y patrimonio' },
    { value: 'ecoturismo', label: 'Ecoturismo' },
    { value: 'buceo', label: 'Buceo y deportes acuáticos' },
    { value: 'senderismo', label: 'Senderismo' },
    { value: 'cultura', label: 'Arte y cultura' },
  ],
};

export async function recommendDestinations(preferences) {
  const apiResult = await fetchRecommendations(preferences);
  const recommendations = apiResult.recommendations || [];

  return recommendations.map((rec) => ({
    id: rec.destination_id || `d${rec.rank}`,
    name: rec.destination,
    icon: '📍',
    match: Math.round(rec.score * 100) || 85,
    category: rec.metadata?.Type || rec.metadata?.type || 'Destino',
    score: rec.score,
    content_score: rec.content_score,
    metadata: rec.metadata,
  }));
}

export const FEATURE_IMPORTANCE = [
  { name: "Demanda Histórica", key: "hist_demand", importance: 0.321 },
  { name: "Incidentes Viales", key: "road_incidents", importance: 0.265 },
  { name: "Capacidad de Rutas", key: "route_capacity", importance: 0.184 },
  { name: "Tasa de Ocupación", key: "occupancy_rate", importance: 0.058 },
  { name: "Tiempo de Espera", key: "waiting_time", importance: 0.045 },
  { name: "Flujo de Pasajeros", key: "passenger_flow", importance: 0.038 },
  { name: "Eficiencia Operativa", key: "op_efficiency", importance: 0.031 },
  { name: "Tipo de Ruta", key: "route_type", importance: 0.029 },
  { name: "Distancia Recorrida", key: "travel_distance", importance: 0.019 },
  { name: "Periodo (30/90 Días)", key: "time_period", importance: 0.010 },
];

export { checkServerHealth } from '../services/api';
