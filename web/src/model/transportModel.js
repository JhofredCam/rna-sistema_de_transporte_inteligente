import {
  fetchDemandForecast,
  fetchDriverClassification,
} from '../services/api';

export const ROUTES = [
  { id: 0, name: 'Ruta A' },
  { id: 1, name: 'Ruta B' },
  { id: 2, name: 'Ruta C' },
  { id: 3, name: 'Ruta D' },
  { id: 4, name: 'Ruta E' },
];

export async function predictDemand(routeId) {
  const apiResult = await fetchDemandForecast(routeId, 30);
  const historyRows = apiResult.historico ?? [];
  const historicalPredRows = apiResult.prediccion_historica ?? [];
  const routeName = apiResult.ruta ?? ROUTES.find((r) => r.id === Number(routeId))?.name ?? 'Ruta A';

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

export const DESTINATIONS = [
  { id: 'd1', name: 'Centro Histórico', icon: '🏛️', match: 95, category: 'Cultura' },
  { id: 'd2', name: 'Malecón Turístico', icon: '🌊', match: 88, category: 'Turismo' },
  { id: 'd3', name: 'Zona Comercial Norte', icon: '🛍️', match: 82, category: 'Compras' },
  { id: 'd4', name: 'Parque Industrial', icon: '🏭', match: 76, category: 'Negocios' },
  { id: 'd5', name: 'Plaza Central', icon: '⛲', match: 91, category: 'Recreación' },
  { id: 'd6', name: 'Terminal de Autobuses', icon: '🚌', match: 73, category: 'Transporte' },
  { id: 'd7', name: 'Hospital General', icon: '🏥', match: 68, category: 'Salud' },
  { id: 'd8', name: 'Ciudad Universitaria', icon: '🎓', match: 85, category: 'Educación' },
  { id: 'd9', name: 'Estadio Olímpico', icon: '🏟️', match: 78, category: 'Deportes' },
  { id: 'd10', name: 'Mercado Municipal', icon: '🏪', match: 80, category: 'Comercio' },
];

export function recommendDestinations() {
  const shuffled = [...DESTINATIONS].sort(() => Math.random() - 0.5);
  const numRecs = 3 + Math.floor(Math.random() * 3);
  return shuffled.slice(0, numRecs).map((d) => ({
    ...d,
    match: Math.max(58, d.match - Math.floor(Math.random() * 18)),
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
