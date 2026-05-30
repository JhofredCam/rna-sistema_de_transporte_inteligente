const raw = import.meta.env.VITE_API_URL || '';
const API_BASE = raw.replace(/\/+$/, '');

export async function fetchDemandPrediction(
  sequence,
  routeId,
  climaId,
  futureFeatures = null,
  futureClimaIds = null,
) {
  const body = {
    sequence,
    route_id: routeId,
    steps: futureClimaIds ? futureClimaIds.length : 1,
  };

  if (futureFeatures && futureClimaIds) {
    body.future_features = futureFeatures;
    body.future_clima_ids = futureClimaIds;
  } else {
    body.clima_id = climaId;
  }

  const res = await fetch(`${API_BASE}/demand/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/`);
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'online';
  } catch {
    return false;
  }
}
