# ChatBot de CV: Comparativa RAG vs Tamaño de Contexto

![Banner](https://github.com/warc0s/Chatbot_CV/blob/dev/extra/banner.png?raw=true)

## Descripción General
Este proyecto implementa y evalúa dos tecnologías avanzadas de chatbot utilizando la información de mi currículum profesional. Ofrece un análisis comparativo entre dos enfoques de inteligencia artificial: RAG (Generación Aumentada por Recuperación) y modelos con gran ventana de contexto.

## ¿Qué es RAG y Tamaño de Contexto?
- **RAG (Generación Aumentada por Recuperación)**: Es una técnica que permite a un modelo de IA buscar información relevante en documentos externos antes de generar una respuesta. Funciona como si el modelo tuviera un asistente que primero busca la información necesaria y luego la utiliza para responder.

- **Tamaño de Contexto**: Representa cuánta información puede "recordar" un modelo de IA durante una conversación. Un modelo con mayor tamaño de contexto puede mantener en memoria más información sin necesidad de buscarla externamente.

## Tecnologías Implementadas

### Enfoque LLM + RAG
- **Modelo**: Llama 3.3 (70B parámetros)
- **Funcionamiento**: El sistema primero busca en mi CV la información relevante a la pregunta y luego genera una respuesta basada en esos datos específicos.
- **Ventajas**: Respuestas precisas y altamente contextualizadas, incluso para información detallada o poco común.

### Enfoque de Gran Ventana de Contexto
- **Modelo**: Gemini Flash
- **Capacidad**: Memoria de 1 millón de tokens (equivalente a cientos de páginas de texto)
- **Ventajas**: Conversaciones fluidas y rápidas sin necesidad de buscar externamente, ya que mantiene todo el CV en memoria.

## Sistema Inteligente de Clasificación

Para optimizar tanto el rendimiento como los costos, se implementa una arquitectura que incluye:

- **Clasificador**: Modelo BERT en español adaptado para clasificar la intención de las preguntas
- **Categorías de Clasificación**:
  1. **Preguntas simples**: Como "¿Qué estudió Marcos?" → Utiliza Gemini Flash Lite (económico)
  2. **Preguntas complejas**: Como "¿Se ajusta el perfil de Marcos a estos requisitos de trabajo?" → Utiliza Gemini Flash (más potente)
  3. **Preguntas fuera de tema**: Como "¿Qué tiempo hace hoy?" → Bloqueadas (evitando uso inadecuado)

## Optimización de Costos y Rendimiento
El sistema dirige inteligentemente cada consulta al modelo más adecuado según su complejidad, garantizando el mejor rendimiento posible mientras controla los costos y previene usos indebidos del sistema.

## Disclaimers Importantes

- **Sistema de clasificación BERT**: Actualmente, el clasificador BERT para detección de intenciones SOLO está implementado para la versión de Gemini Flash.

- **Estado del enfoque LLM+RAG**: La implementación con Llama 3.3 y RAG todavía NO está desarrollada. Este enfoque se implementará en futuras actualizaciones del proyecto utilizando el framework LlamaIndex para la parte de recuperación y generación.

- **Fase actual**: El proyecto se encuentra en fase de desarrollo y evaluación, por lo que algunas funcionalidades pueden estar sujetas a cambios significativos.

## Licencia
Licenciado bajo Apache License 2.0. Consulte el archivo LICENSE para más detalles.
