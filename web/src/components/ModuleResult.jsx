import { createElement, forwardRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Camera,
  MapPin,
  Star,
  BarChart3,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';

const ChartTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass rounded-xl p-3 text-sm border border-brand-500/30">
        <p className="font-semibold text-white mb-1">Día {label}</p>
        {payload.map((entry) => (
          entry.value != null && (
            <p key={entry.name} style={{ color: entry.color }} className="text-xs">
              {entry.name === 'real' ? 'Demanda Real' : 'Predicción'}: {entry.value.toLocaleString()} pax
            </p>
          )
        ))}
      </div>
    );
  }
  return null;
};

const MetricCard = ({ icon, label, value, color }) => (
  <div className="glass rounded-2xl p-5 flex items-center gap-4 hover:bg-white/5 transition-all">
    <div
      className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
      style={{ backgroundColor: `${color}20` }}
    >
      {createElement(icon, { className: 'w-6 h-6', style: { color } })}
    </div>
    <div>
      <p className="text-xs text-surface-200/50 font-medium uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-black text-white mt-0.5">{value}</p>
    </div>
  </div>
);

const DemandResult = ({ result }) => {
  if (!result) return null;
  const { days, rmse, mae, routeName } = result;
  const splitDay = days.findIndex((d) => d.real == null && d.predicted != null);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="glass rounded-3xl p-8 glow-brand">
        <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-brand-400" />
          Predicción de Demanda: {routeName}
        </h3>
        <p className="text-sm text-surface-200/40 mb-6">
          60 días históricos + pronóstico 30 días — la línea naranja en los primeros 60 días compara la predicción del modelo vs el valor real
        </p>

        <div className="h-72 sm:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={days} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="realGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#83a598" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#83a598" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#fe8019" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#fe8019" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(168,153,132,0.15)" vertical={false} />
              {splitDay > 0 && (
                <ReferenceLine x={splitDay + 0.5} stroke="#fe8019" strokeDasharray="6 3" strokeWidth={2} label={{ value: 'Pronóstico →', position: 'insideTopRight', fill: '#fe8019', fontSize: 12 }} />
              )}
              <XAxis
                dataKey="day"
                tick={{ fill: '#ebdbb2', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(168,153,132,0.3)' }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 3500]}
                tick={{ fill: '#ebdbb2', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 12, color: '#ebdbb2' }}
                formatter={(value) => (
                  <span style={{ color: '#ebdbb2' }}>
                    {value === 'real' ? 'Demanda Real' : 'Predicción'}
                  </span>
                )}
              />
              <Area
                type="monotone"
                dataKey="real"
                stroke="#83a598"
                strokeWidth={2}
                fill="url(#realGrad)"
                name="real"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="predicted"
                stroke="#fe8019"
                strokeWidth={2}
                strokeDasharray="5 5"
                fill="url(#predGrad)"
                name="predicted"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <MetricCard
          icon={Target}
          label="RMSE de Evaluación"
          value={rmse == null ? 'Ver entrenamiento' : rmse.toLocaleString()}
          color="#83a598"
        />
        <MetricCard
          icon={Activity}
          label="MAE de Evaluación"
          value={mae == null ? 'Ver entrenamiento' : mae.toLocaleString()}
          color="#8ec07c"
        />
      </div>
    </div>
  );
};

const ClassificationResult = ({ result }) => {
  if (!result) return null;
  const { label, icon, color, confidence, filename, preventiveMeasure, probabilities } = result;
  const confidencePercent = Math.round(confidence * 100);
  const probabilityRows = Object.entries(probabilities ?? {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className={`glass rounded-3xl p-8`} style={{ boxShadow: `0 0 30px ${color}15, 0 0 60px ${color}08` }}>
        <div className="flex flex-col items-center text-center gap-6">
          <div
            className="w-24 h-24 rounded-3xl flex items-center justify-center text-5xl"
            style={{ backgroundColor: `${color}20` }}
          >
            {icon}
          </div>

          <div>
            <div
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold mb-3"
              style={{
                backgroundColor: `${color}15`,
                color,
                border: `1px solid ${color}30`,
              }}
            >
              <Camera className="w-3.5 h-3.5" />
              Resultado de Clasificación
            </div>
            <h3 className="text-3xl font-black text-white">{label}</h3>
          </div>

          <div className="w-full max-w-md space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-surface-200/60">Confianza del Modelo</span>
              <span className="font-bold text-white">{confidencePercent}%</span>
            </div>
            <div className="h-4 rounded-full bg-surface-900/60 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000"
                style={{
                  width: `${confidencePercent}%`,
                  background: `linear-gradient(90deg, ${color}88, ${color})`,
                }}
              />
            </div>
          </div>

          <p className="text-sm text-surface-200/40 max-w-md">
            El modelo CNN ha analizado la imagen del conductor y asignado la categoría
            con un nivel de confianza del {confidencePercent}%.
          </p>

          {filename && (
            <p className="text-xs text-surface-200/30 max-w-md">
              Archivo analizado: {filename}
            </p>
          )}

          {preventiveMeasure && (
            <div className="w-full max-w-2xl rounded-2xl border border-surface-700/50 bg-surface-900/40 p-4 text-left">
              <p className="text-xs text-surface-200/50 font-medium uppercase tracking-wider mb-2">
                Medida preventiva sugerida
              </p>
              <p className="text-sm text-surface-200/80">{preventiveMeasure}</p>
            </div>
          )}
        </div>
      </div>

      {probabilityRows.length > 0 && (
        <div className="glass rounded-3xl p-6">
          <h4 className="text-sm font-bold text-white mb-4">Probabilidades por clase</h4>
          <div className="space-y-3">
            {probabilityRows.map(([className, value]) => (
              <div key={className} className="space-y-1.5">
                <div className="flex justify-between gap-4 text-xs">
                  <span className="text-surface-200/70">{className}</span>
                  <span className="font-semibold text-white">{Math.round(value * 100)}%</span>
                </div>
                <div className="h-2 rounded-full bg-surface-900/60 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.round(value * 100)}%`,
                      background: `linear-gradient(90deg, ${color}88, ${color})`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const RecommendationResult = ({ result }) => {
  if (!result) return null;
  const { destinations, clientId } = result;

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div className="glass rounded-3xl p-8 glow-brand">
        <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
          <MapPin className="w-5 h-5 text-brand-400" />
          Destinos Recomendados para {clientId}
        </h3>
        <p className="text-sm text-surface-200/40 mb-6">
          Basado en filtrado colaborativo e historial de viajes
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {destinations.map((dest, idx) => (
            <div
              key={dest.id}
              className="glass rounded-2xl p-5 hover:bg-white/5 transition-all duration-300
                hover:border-brand-500/30 hover:translate-y-[-2px] group animate-fade-in-up"
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              <div className="text-5xl mb-4 group-hover:scale-110 transition-transform duration-300 block text-center">
                {dest.icon}
              </div>
              <h4 className="font-bold text-white text-center mb-3">{dest.name}</h4>

              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-surface-200/50">Coincidencia</span>
                  <span className="font-semibold text-white">{dest.match}%</span>
                </div>
                <div className="h-2 rounded-full bg-surface-900/60 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${dest.match}%`,
                      background: `linear-gradient(90deg, #83a59888, #83a598)`,
                    }}
                  />
                </div>
              </div>

              <div className="mt-3 flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-wider text-surface-200/30 font-medium">
                  {dest.category}
                </span>
                <div className="flex items-center gap-1">
                  <Star className="w-3 h-3 text-status-warn" />
                  <span className="text-xs text-status-warn font-semibold">{dest.match}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const ModuleResult = forwardRef(({ demandResult, classificationResult, recommendationResult, moduleTab }, ref) => {
  const hasDemand = moduleTab === 'demand' && demandResult;
  const hasClassification = moduleTab === 'classification' && classificationResult;
  const hasRecommendation = moduleTab === 'recommendation' && recommendationResult;

  if (!hasDemand && !hasClassification && !hasRecommendation) return null;

  return (
    <div ref={ref}>
      {hasDemand && <DemandResult result={demandResult} />}
      {hasClassification && <ClassificationResult result={classificationResult} />}
      {hasRecommendation && <RecommendationResult result={recommendationResult} />}
    </div>
  );
});

ModuleResult.displayName = 'ModuleResult';
export default ModuleResult;
