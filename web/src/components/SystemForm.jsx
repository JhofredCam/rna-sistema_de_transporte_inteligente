import { createElement, useState, useRef, useCallback } from 'react';
import {
  Calculator,
  RotateCcw,
  TrendingUp,
  Camera,
  MapPin,
  Upload,
  FileImage,
} from 'lucide-react';
import { ROUTES, DRIVER_CLASSES } from '../model/transportModel';

const TABS = [
  { id: 'demand', label: 'Predicción de Demanda', icon: TrendingUp },
  { id: 'classification', label: 'Clasificación de Conducción', icon: Camera },
  { id: 'recommendation', label: 'Recomendación de Destinos', icon: MapPin },
];

export default function SystemForm({
  onPredictDemand,
  onClassifyDriver,
  onRecommendDestinations,
  isProcessing,
  activeModule,
}) {
  const [tab, setTab] = useState(activeModule || 'demand');
  const [selectedRoute, setSelectedRoute] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [clientId, setClientId] = useState('');
  const fileInputRef = useRef(null);

  const handleTabChange = useCallback(
    (newTab) => {
      setTab(newTab);
      if (onPredictDemand) onPredictDemand(null, newTab);
    },
    [onPredictDemand]
  );

  const handleDemandSubmit = (e) => {
    e.preventDefault();
    if (onPredictDemand) onPredictDemand(selectedRoute, 'demand');
  };

  const handleDemandReset = () => {
    setSelectedRoute(0);
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0] || e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleClassify = (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    if (onClassifyDriver) onClassifyDriver(selectedFile, 'classification');
  };

  const handleClassifyReset = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRecommend = (e) => {
    e.preventDefault();
    if (!clientId.trim()) return;
    if (onRecommendDestinations) onRecommendDestinations(clientId.trim(), 'recommendation');
  };

  const handleRecommendReset = () => setClientId('');

  return (
    <div className="space-y-6">
      <div className="flex border-b border-surface-700/50">
        {TABS.map(({ id, label, icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => handleTabChange(id)}
            className={`flex items-center gap-2 px-4 sm:px-6 py-3 text-sm font-medium transition-all cursor-pointer border-b-2 -mb-px ${
              tab === id
                ? 'border-brand-500 text-brand-300'
                : 'border-transparent text-surface-200/60 hover:text-surface-200 hover:border-surface-200/30'
            }`}
          >
            {createElement(icon, { className: 'w-4 h-4' })}
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {tab === 'demand' && (
        <form onSubmit={handleDemandSubmit} className="space-y-6 pt-2">
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-surface-200">
              Seleccionar Ruta de Transporte
            </label>
            <select
              value={selectedRoute}
              onChange={(e) => setSelectedRoute(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-surface-900/60 border border-surface-700/50
                text-white focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500
                transition-all hover:border-surface-200/30 cursor-pointer appearance-none"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%2383a598' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 12px center',
              }}
            >
              {ROUTES.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 pt-2">
            <button
              type="submit"
              disabled={isProcessing}
              className={`btn-primary flex-1 flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-semibold text-lg transition-all duration-300 cursor-pointer ${
                isProcessing
                  ? 'bg-brand-700/50 text-brand-200/50 cursor-wait'
                  : 'bg-gradient-to-r from-brand-600 to-brand-500 text-white hover:from-brand-500 hover:to-brand-400 hover:shadow-xl hover:shadow-brand-500/20 hover:scale-[1.02]'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-brand-300/30 border-t-brand-300 rounded-full animate-spin" />
                  Generando Predicción...
                </>
              ) : (
                <>
                  <TrendingUp className="w-5 h-5" />
                  Generar Predicción a 30 Días
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleDemandReset}
              className="flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-medium
                border border-surface-700/50 text-surface-200 hover:bg-white/5 hover:border-surface-200/30 transition-all cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
              Reiniciar
            </button>
          </div>
        </form>
      )}

      {tab === 'classification' && (
        <form onSubmit={handleClassify} className="space-y-6 pt-2">
          <div
            onDrop={handleFileDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all
              ${
                dragOver
                  ? 'border-brand-500 bg-brand-500/10'
                  : selectedFile
                    ? 'border-brand-500/40 bg-brand-500/5'
                    : 'border-surface-700/50 hover:border-surface-200/30 hover:bg-white/5'
              }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileDrop}
            />
            {selectedFile ? (
              <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-2xl bg-brand-600/20 flex items-center justify-center">
                  <FileImage className="w-8 h-8 text-brand-400" />
                </div>
                <div>
                  <p className="text-white font-medium">{selectedFile.name}</p>
                  <p className="text-xs text-surface-200/40 mt-1">
                    Haz clic o arrastra para cambiar la imagen
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-2xl bg-surface-700/30 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-surface-200/40" />
                </div>
                <div>
                  <p className="text-surface-200/60 font-medium">
                    Arrastra una imagen del conductor aquí
                  </p>
                  <p className="text-xs text-surface-200/30 mt-1">
                    o haz clic para seleccionar un archivo
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {DRIVER_CLASSES.map((dc) => (
              <div
                key={dc.id}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-900/40 text-xs text-surface-200/60"
              >
                <span>{dc.icon}</span>
                <span>{dc.label}</span>
              </div>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            <button
              type="submit"
              disabled={isProcessing || !selectedFile}
              className={`btn-primary flex-1 flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-semibold text-lg transition-all duration-300 cursor-pointer ${
                isProcessing || !selectedFile
                  ? 'bg-brand-700/50 text-brand-200/50 cursor-wait'
                  : 'bg-gradient-to-r from-brand-600 to-brand-500 text-white hover:from-brand-500 hover:to-brand-400 hover:shadow-xl hover:shadow-brand-500/20 hover:scale-[1.02]'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-brand-300/30 border-t-brand-300 rounded-full animate-spin" />
                  Analizando...
                </>
              ) : (
                <>
                  <Camera className="w-5 h-5" />
                  Clasificar Conductor
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleClassifyReset}
              className="flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-medium
                border border-surface-700/50 text-surface-200 hover:bg-white/5 hover:border-surface-200/30 transition-all cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
              Reiniciar
            </button>
          </div>
        </form>
      )}

      {tab === 'recommendation' && (
        <form onSubmit={handleRecommend} className="space-y-6 pt-2">
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-surface-200">
              ID de Cliente
            </label>
            <input
              type="text"
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="Ej: 1"
              className="w-full px-4 py-3 rounded-xl bg-surface-900/60 border border-surface-700/50
                text-white placeholder-surface-200/30 focus:outline-none focus:ring-2
                focus:ring-brand-500/50 focus:border-brand-500 transition-all hover:border-surface-200/30"
            />
            <p className="text-xs text-surface-200/30">
              Ingresa un identificador único para generar recomendaciones personalizadas.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 pt-2">
            <button
              type="submit"
              disabled={isProcessing || !clientId.trim()}
              className={`btn-primary flex-1 flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-semibold text-lg transition-all duration-300 cursor-pointer ${
                isProcessing || !clientId.trim()
                  ? 'bg-brand-700/50 text-brand-200/50 cursor-wait'
                  : 'bg-gradient-to-r from-brand-600 to-brand-500 text-white hover:from-brand-500 hover:to-brand-400 hover:shadow-xl hover:shadow-brand-500/20 hover:scale-[1.02]'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-brand-300/30 border-t-brand-300 rounded-full animate-spin" />
                  Generando Recomendaciones...
                </>
              ) : (
                <>
                  <MapPin className="w-5 h-5" />
                  Recomendar Destinos
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handleRecommendReset}
              className="flex items-center justify-center gap-2 px-6 py-4 rounded-xl font-medium
                border border-surface-700/50 text-surface-200 hover:bg-white/5 hover:border-surface-200/30 transition-all cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
              Reiniciar
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
