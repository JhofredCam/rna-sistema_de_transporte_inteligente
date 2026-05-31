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

export async function fetchDemandForecast(routeId, steps = 30) {
  const res = await fetch(`${API_BASE}/demand/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      route_id: Number(routeId),
      steps,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function fetchDemandMetadata() {
  const res = await fetch(`${API_BASE}/demand/metadata`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function fetchDriverClassification(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/distraction/predict`, {
    method: 'POST',
    body: formData,
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

export async function fetchRecommendations(userId, topK = 6, excludeSeen = true) {
  const params = new URLSearchParams({
    user_id: userId,
    top_k: String(topK),
    exclude_seen: String(excludeSeen),
  });
  const res = await fetch(`${API_BASE}/recommender/recommend?${params}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

export async function fetchRecommenderInfo() {
  const res = await fetch(`${API_BASE}/recommender/info`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) return null;
  return res.json();
}
