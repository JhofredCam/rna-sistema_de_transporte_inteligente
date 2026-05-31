import { Sparkles } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import AboutModel from './components/AboutModel';
import Footer from './components/Footer';
import Hero from './components/Hero';
import ModuleResult from './components/ModuleResult';
import Navbar from './components/Navbar';
import ReadmeViewer from './components/ReadmeViewer';
import Resources from './components/Resources';
import SystemForm from './components/SystemForm';
import { checkServerHealth, classifyDriver, predictDemand, recommendDestinations } from './model/transportModel';

export default function App() {
  const [demandResult, setDemandResult] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [recommendationResult, setRecommendationResult] = useState(null);
  const [moduleTab, setModuleTab] = useState('demand');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const resultRef = useRef(null);
  const [showDocs, setShowDocs] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking');

  useEffect(() => {
    checkServerHealth().then((online) => {
      setServerStatus(online ? 'online' : 'offline');
    });
  }, []);

  const handlePredictDemand = useCallback(async (routeId, tab) => {
    setModuleTab(tab || 'demand');
    if (routeId === null || routeId === undefined) return;
    setIsProcessing(true);
    setError(null);
    try {
      const result = await predictDemand(routeId);
      setDemandResult(result);
    } catch (err) {
      setDemandResult(null);
      setError(err?.message || 'Error al conectar con el servidor');
    }
    setIsProcessing(false);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  const handleClassifyDriver = useCallback(async (file, tab) => {
    setModuleTab(tab || 'classification');
    setIsProcessing(true);
    try {
      const result = await classifyDriver(file);
      setClassificationResult(result);
    } catch {
      setClassificationResult(null);
    } finally {
      setIsProcessing(false);
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, []);

  const handleRecommendDestinations = useCallback(async (clientId, tab, topK = 6, excludeSeen = true) => {
    setModuleTab(tab || 'recommendation');
    setIsProcessing(true);
    setError(null);
    setRecommendationResult(null);
    try {
      const destinations = await recommendDestinations(clientId, topK, excludeSeen);
      setRecommendationResult({ destinations, clientId });
    } catch (err) {
      setRecommendationResult(null);
      setError(err?.message || 'Error al obtener recomendaciones');
    }
    setIsProcessing(false);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  const handleToggleDocs = useCallback(() => {
    setShowDocs((prev) => {
      const newState = !prev;
      if (newState) {
        setTimeout(() => {
          document.getElementById('documentacion')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
      }
      return newState;
    });
  }, []);

  const handleShowDocs = useCallback(() => {
    setShowDocs(true);
    setTimeout(() => {
      document.getElementById('documentacion')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }, []);

  return (
    <div className="bg-animated min-h-screen">
      <Navbar onShowDocs={handleShowDocs} />

      <Hero serverStatus={serverStatus} />

      <section id="modulos" className="py-24 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-light mb-4">
              <Sparkles className="w-4 h-4 text-brand-400" />
              <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
                Herramientas
              </span>
            </div>
            <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
              Funcionalidades del Sistema
            </h2>
            <p className="text-surface-200/60 max-w-xl mx-auto">
              Explora los tres módulos principales del sistema: predicción de demanda,
              clasificación de conductores y recomendación de destinos personalizados.
            </p>
          </div>

          <div className="glass rounded-3xl p-6 sm:p-8 glow-brand mb-8">
            <SystemForm
              onPredictDemand={handlePredictDemand}
              onClassifyDriver={handleClassifyDriver}
              onRecommendDestinations={handleRecommendDestinations}
              isProcessing={isProcessing}
              activeModule={moduleTab}
            />
          </div>

          {error && (
            <div className="mb-6 p-4 rounded-2xl bg-red-900/20 border border-red-500/30 text-red-300 text-sm">
              {error}
            </div>
          )}
          <ModuleResult
            ref={resultRef}
            demandResult={demandResult}
            classificationResult={classificationResult}
            recommendationResult={recommendationResult}
            moduleTab={moduleTab}
          />


        </div>
      </section>

      <AboutModel />
      <Resources onToggleDocs={handleToggleDocs} showDocs={showDocs} />
      {showDocs && <ReadmeViewer />}
      <Footer />
    </div>
  );
}
