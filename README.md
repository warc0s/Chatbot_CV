# 🤖 ChatBot CV: RAG vs Ventana de Contexto

![Banner](https://github.com/warc0s/Chatbot_CV/blob/main/extra/banner.png?raw=true)

## 📋 Índice
- [Descripción General](#-descripción-general)
- [Arquitectura Dual: RAG vs Ventana de Contexto](#-arquitectura-dual-rag-vs-ventana-de-contexto)
  - [¿Qué es RAG?](#qué-es-rag)
  - [¿Qué es una Gran Ventana de Contexto?](#qué-es-una-gran-ventana-de-contexto)
  - [Comparación de Enfoques](#comparación-de-enfoques)
- [Implementaciones Técnicas](#-implementaciones-técnicas)
  - [Enfoque 1: LlamaIndex + Llama 3.3 70B](#enfoque-1-llamaindex--llama-33-70b)
  - [Enfoque 2: Gemini Flash con Contexto Completo](#enfoque-2-gemini-flash-con-contexto-completo)
- [Sistema Inteligente de Clasificación](#-sistema-inteligente-de-clasificación)
- [Arquitectura y Optimización](#%E2%9A%99%EF%B8%8F-arquitectura-y-optimización)
- [Frontend y Experiencia de Usuario](#-frontend-y-experiencia-de-usuario)
- [Tecnologías y Frameworks](#%F0%9F%9B%A0%EF%B8%8F-tecnologías-y-frameworks)
- [Instalación y Uso](#-instalación-y-uso)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## 📋 Descripción General

Este proyecto implementa y compara dos arquitecturas avanzadas de chatbot aplicadas a la consulta de información curricular. El objetivo es analizar las diferencias en rendimiento, precisión y experiencia de usuario entre:

1. **Generación Aumentada por Recuperación (RAG)** con Llama 3.3 y LlamaIndex
2. **Modelo con Gran Ventana de Contexto** usando Gemini Flash

Ambos chatbots proporcionan respuestas detalladas sobre mi experiencia profesional, habilidades y formación, pero utilizan mecanismos fundamentalmente diferentes para acceder y procesar la información. Este proyecto ofrece así un interesante caso de estudio sobre los trade-offs entre diferentes arquitecturas de IA conversacional.

## 🧠 Arquitectura Dual: RAG vs Ventana de Contexto

### ¿Qué es RAG?

**RAG (Retrieval Augmented Generation)** es una arquitectura que combina:

1. **Recuperación (Retrieval)**: El sistema busca en tiempo real información relevante en una base de conocimientos (en este caso, mi CV indexado con embeddings vectoriales).

2. **Generación Aumentada**: El modelo de lenguaje (LLM) utiliza la información recuperada para generar respuestas más precisas y fundamentadas en datos concretos.

RAG funciona como un sistema de dos etapas:
- Primero convierte la pregunta del usuario en una consulta vectorial
- Luego busca los fragmentos más relevantes del CV mediante similitud semántica
- Finalmente, proporciona estos fragmentos como contexto adicional al LLM para generar la respuesta

### ¿Qué es una Gran Ventana de Contexto?

La **ventana de contexto** determina cuánta información puede "ver" un modelo en una sola conversación. Modelos como Gemini Flash poseen una ventana de contexto excepcionalmente grande (hasta 1 millón de tokens), lo que permite:

- Cargar documentos completos (como mi CV) directamente en el prompt
- Mantener toda la conversación previa dentro del contexto
- Responder basándose en una comprensión holística del documento sin necesidad de recuperación externa

En este enfoque, el modelo tiene acceso constante a toda la información, similar a como un humano tendría un documento completo frente a él durante toda la conversación.

### Comparación de Enfoques

| Aspecto | Enfoque RAG | Enfoque Gran Ventana |
|---------|-------------|----------------------|
| **Precisión** | Precisión variable dependiente de la calidad de recuperación y estrategia de chunking | Comprensión completa del documento con visión global |
| **Flexibilidad** | Puede acceder a grandes volúmenes de información externa | Limitado estrictamente a lo que se cargó en la ventana de contexto |
| **Consumo de recursos** | Uso eficiente de tokens en el LLM, pero requiere infraestructura para vectorización e indexación | Consumo elevado de tokens en cada consulta independientemente de su complejidad |
| **Velocidad** | Mayor latencia debido al proceso de búsqueda + generación | Respuesta más directa y rápida al tener toda la información pre-cargada |
| **Escalabilidad** | Puede escalar prácticamente sin límites en volumen de información | Estrictamente limitado por el tamaño máximo de la ventana de contexto |
| **Dependencia de chunking** | Altamente dependiente de la estrategia de segmentación; chunks mal diseñados pueden omitir información relevante | No requiere chunking, eliminando este posible problema |
| **Adaptabilidad** | Permite actualizar la base de conocimientos sin reentrenamiento | Requiere regenerar el prompt completo para actualizar información |

## 💻 Implementaciones Técnicas

### Enfoque 1: LlamaIndex + Llama 3.3 70B

Esta implementación utiliza:

- **Modelo base**: Llama 3.3 70B de Groq
- **Framework RAG**: LlamaIndex para indexación y recuperación
- **Embeddings**: BGE-M3 (BAAI) a través de DeepInfra para crear representaciones vectoriales del CV
- **Almacenamiento**: Índice vectorial persistente para optimizar rendimiento
- **Backend**: Flask con arquitectura RESTful
- **Manejo de sesiones**: Sistema de historial de conversación con contexto persistente

El flujo de funcionamiento:
1. La consulta del usuario se vectoriza
2. Se recuperan los fragmentos más relevantes del CV
3. Se combina la consulta, fragmentos relevantes e historial de conversación
4. Se genera una respuesta contextualizada con el modelo Llama 3.3 70B

### Enfoque 2: Gemini Flash con Contexto Completo

Esta versión aprovecha la enorme ventana de contexto:

- **Modelo principal**: Gemini 2.0 Flash (1M tokens de contexto)
- **Modelo alternativo**: Gemini 2.0 Flash-Lite (para consultas sencillas)
- **Prompt Engineering**: CV completo incluido en el contexto inicial
- **Clasificador**: BERT en español para categorizar consultas
- **Integración**: LiteLLM para gestión de API y solicitudes
- **Gestión de memoria**: Optimización del historial de conversación

Funcionamiento:
1. El CV completo se carga en el contexto del modelo (System Prompt)
2. BERT clasifica la consulta por complejidad
3. Se selecciona el modelo apropiado (Flash o Flash-Lite)
4. Se genera la respuesta con todo el contexto disponible

## 🔍 Sistema Inteligente de Clasificación

El sistema BERT implementado en la versión Gemini clasifica las consultas en tres categorías:

1. **Consultas simples** (0): Preguntas directas sobre información básica del CV
   - Ejemplo: "¿Dónde estudió Marcos?"
   - Procesadas por: Gemini Flash-Lite (más económico y rápido)

2. **Consultas complejas** (1): Preguntas que requieren análisis, razonamiento o síntesis
   - Ejemplo: "¿Cómo se complementan sus habilidades técnicas y de gestión?"
   - Procesadas por: Gemini Flash (más potente y con mayor capacidad de razonamiento)

3. **Consultas fuera de ámbito** (2): Preguntas no relacionadas con el CV
   - Ejemplo: "¿Cómo funciona la fusión nuclear?"
   - Resultado: Bloqueadas para evitar uso inadecuado del sistema

Esta clasificación inteligente permite:
- Optimizar costos dirigiendo consultas al modelo más adecuado
- Mejorar tiempos de respuesta para preguntas simples
- Mantener el enfoque del chatbot en su propósito específico
- Prevenir posibles abusos o desvíos del tema principal

## ⚙️ Arquitectura y Optimización

Ambas implementaciones comparten elementos arquitectónicos clave:

- **Separación backend/frontend**: API REST desacoplada de la interfaz de usuario
- **Gestión de sesiones**: Identificadores únicos para mantener conversaciones coherentes
- **Persistencia**: Almacenamiento eficiente de índices (en LlamaIndex) y sesiones
- **Concurrencia**: Control de solicitudes simultáneas para evitar sobrecarga
- **Caducidad**: Limpieza automática de sesiones inactivas
- **Prompt Engineering**: Instrucciones detalladas para garantizar respuestas profesionales y precisas

Optimizaciones específicas:
- **RAG**: Chunking personalizado, caching de embeddings
- **Contexto**: Gestión eficiente de tokens, historial selectivo, plantillas de prompt optimizadas

## 🎨 Frontend y Experiencia de Usuario

Ambas versiones presentan interfaces modernas y responsivas:

- **UI Adaptativa**: Diseño responsive con Tailwind CSS
- **Animaciones**: Transiciones suaves para mejorar la experiencia
- **Mensajes de Estado**: Indicadores durante el procesamiento con etapas descriptivas
- **Markdown**: Soporte completo para respuestas formateadas con listas, enlaces, etc.
- **Historial**: Persistencia de conversaciones entre sesiones
- **Accesibilidad**: Contraste adecuado y estructuras semánticas
- **Mobile-first**: Optimizado para dispositivos móviles y escritorio

## 🛠️ Tecnologías y Frameworks

### Modelos y APIs
- **Llama 3.3 70B**: A través de Groq API
- **Gemini 2.0 Flash/Flash-Lite**: Google AI API
- **BGE-M3**: Embeddings vía DeepInfra
- **BERT**: Fine-tuned para clasificación de intenciones

### Frameworks y Librerías
- **LlamaIndex**: Framework para implementación RAG
- **Flask**: Backend ligero y eficiente
- **Tailwind CSS**: Estilizado moderno y responsive
- **LiteLLM**: Abstracción para APIs de cualquier proveedor de LLM
- **Marcado/Renderizado**: Marked.js para procesamiento de markdown
- **LocalStorage**: Persistencia en el lado cliente

## 📦 Instalación y Uso

### Requisitos
- Python 3.10+
- API Keys: Groq, Google AI, DeepInfra

### Configuración del Backend
*Instrucciones de instalación:*

*# Clonar el repositorio*
*git clone https://github.com/warc0s/Chatbot_CV.git*
*cd Chatbot_CV*

*# Crear entorno virtual*
*python -m venv venv*
*source venv/bin/activate  # En Windows: venv\Scripts\activate*

*# Instalar dependencias*
*pip install -r requirements.txt*

*# Configurar variables de entorno*
*# Editar .env con tus API keys*

*# Iniciar servidor*
*python app.py*

### Uso del Frontend
El frontend estará disponible en `http://localhost:5000` tras iniciar el servidor.

## 👥 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva característica'`)
4. Sube los cambios (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📜 Licencia

Este proyecto está licenciado bajo Apache License 2.0. Consulta el archivo [LICENSE](LICENSE) para más detalles.
