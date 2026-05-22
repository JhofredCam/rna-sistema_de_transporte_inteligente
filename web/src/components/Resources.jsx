/**
 * ============================================
 * Resources - Sección de recursos y enlaces
 * ============================================
 * Enlaces a documentación técnica, reporte de 11 capítulos,
 * repositorio y recursos del proyecto de transporte inteligente.
 */
import { FileText, PlayCircle, Code, ExternalLink } from 'lucide-react';

const resources = [
  {
    id: 'docs',
    icon: FileText,
    title: 'Documentación Técnica',
    description:
      'Explora el reporte técnico completo del proyecto con los 11 capítulos: portada, resumen, metodología, desarrollo por módulo y resultados.',
    link: '#documentacion',
    linkLabel: 'Ver Documentación',
    color: '#83a598',
    badge: 'PDF',
  },
  {
    icon: Code,
    title: 'Código Fuente',
    description:
      'Repositorio con los notebooks de entrenamiento (demanda, conducción, recomendación), modelos exportados y código de la aplicación web.',
    link: 'https://github.com/JhofredCam/rna-sistema_de_transporte_inteligente',
    linkLabel: 'Ver Repositorio',
    color: '#a89984',
    badge: 'GitHub',
  },
];

export default function Resources({ onToggleDocs, showDocs }) {
  return (
    <section id="recursos" className="py-24 px-4 sm:px-6">
      <div className="max-w-5xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full glass-light mb-4">
            <FileText className="w-4 h-4 text-brand-400" />
            <span className="text-xs font-semibold text-brand-300 uppercase tracking-wider">
              Recursos
            </span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-black text-white mb-4">
            Reporte Técnico y Código
          </h2>
          <p className="text-surface-200/60 max-w-xl mx-auto">
            Accede al reporte técnico completo de 11 capítulos y al repositorio con
            notebooks, modelos y código fuente del proyecto.
          </p>
        </div>

        {/* Resource cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {resources.map(({ id, icon: Icon, title, description, link, linkLabel, color, badge }) => (
            <a
              key={title}
              href={link}
              onClick={(e) => {
                if (id === 'docs') {
                  e.preventDefault();
                  onToggleDocs();
                }
              }}
              className="glass rounded-2xl p-6 group hover:bg-white/5 transition-all duration-300
                hover:border-brand-500/30 hover:translate-y-[-4px] hover:shadow-xl hover:shadow-brand-500/10 block cursor-pointer"
            >
              <div className="flex items-center justify-between mb-4">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center
                    group-hover:scale-110 transition-transform duration-300"
                  style={{ backgroundColor: `${color}15` }}
                >
                  <Icon className="w-6 h-6" style={{ color }} />
                </div>
                <span
                  className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full"
                  style={{ backgroundColor: `${color}15`, color }}
                >
                  {badge}
                </span>
              </div>

              <h3 className="font-bold text-white mb-2 group-hover:text-brand-300 transition-colors">
                {title}
              </h3>
              <p className="text-sm text-surface-200/60 leading-relaxed mb-4">
                {description}
              </p>

              <div className="flex items-center gap-1.5 text-sm font-medium text-brand-400 
                group-hover:text-brand-300 transition-colors">
                {id === 'docs' && showDocs ? 'Ocultar Documentación' : linkLabel}
                <ExternalLink className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
