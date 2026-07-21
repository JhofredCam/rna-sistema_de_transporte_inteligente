/**
 * ============================================
 * Hero - Sección principal de bienvenida
 * ============================================
 * Primer bloque visual que el usuario ve.
 * Incluye CTA para explorar el sistema de transporte inteligente.
 */
import { ArrowDown, Shield, Brain } from 'lucide-react';
import { createElement } from 'react';

export default function Hero() {
  const scrollToModules = () => {
    document.getElementById('modulos')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section
      id="inicio"
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Decorative floating orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-600/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-brand-500/8 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 w-72 h-72 bg-brand-400/6 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 text-center">
        {/* Main heading */}
        <h1 className="text-4xl sm:text-5xl md:text-7xl font-black tracking-tight mb-6 animate-fade-in-up">
          <span className="text-white">Sistema Inteligente </span>
          <span className="text-shine">Integrado</span>
          <br />
          <span className="text-white">para la </span>
          <span className="bg-gradient-to-r from-brand-400 to-brand-600 bg-clip-text text-transparent">
            Empresa de Transporte
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg sm:text-xl text-surface-200/80 max-w-2xl mx-auto mb-10 animate-fade-in-up-delay-1 leading-relaxed">
          Sistema basado en aprendizaje profundo para mejorar la eficiencia operativa,
          seguridad vial y experiencia del usuario mediante predicción, clasificación y recomendación.
        </p>

        {/* CTA Button */}
        <div className="animate-fade-in-up-delay-2">
          <button
            onClick={scrollToModules}
            className="btn-primary inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-brand-600 to-brand-500 
            text-white font-semibold text-lg rounded-2xl hover:from-brand-500 hover:to-brand-400 
            transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-brand-500/25 cursor-pointer"
          >
            Explorar el Sistema
            <ArrowDown className="w-5 h-5 animate-bounce" />
          </button>
        </div>

        {/* Stats cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-16 animate-fade-in-up-delay-3">
          {[
            { icon: Shield, label: 'Predicción Demanda', desc: 'Series de Tiempo' },
            { icon: Brain, label: 'Clasificación Conductores', desc: 'Visión por Computadora' },
          ].map(({ icon, label, desc }) => (
            <div
              key={label}
              className="glass rounded-2xl p-6 flex items-center gap-4 hover:bg-white/5 transition-all group"
            >
              <div className="w-12 h-12 rounded-xl bg-brand-600/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                {createElement(icon, { className: 'w-6 h-6 text-brand-400' })}
              </div>
              <div className="text-left">
                <p className="font-semibold text-white">{label}</p>
                <p className="text-sm text-surface-200/60">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
