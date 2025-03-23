import os
import sys
import threading
import logging
import re
import unicodedata
import torch
import uuid
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from transformers import BertForSequenceClassification, BertTokenizer
import litellm

# -------------------------- Configuración de logging --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------- Integración del BERT para clasificación del prompt --------------------------
def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

bert_model = BertForSequenceClassification.from_pretrained("prompt_analysis")
bert_tokenizer = BertTokenizer.from_pretrained("prompt_analysis")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def clasificar_dificultad(texto):
    texto_limpio = clean_text(texto)
    inputs = bert_tokenizer(texto_limpio, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediccion = torch.argmax(logits, dim=1).item()
    return prediccion

# ------------- Sistema de memoria para conversaciones -------------
conversation_history = {}  # {session_id: [lista de mensajes]}
last_activity = {}  # {session_id: timestamp}
SESSION_TIMEOUT = timedelta(hours=12)  # Sesiones inactivas por más de 12 horas se eliminarán

def clean_inactive_sessions():
    """Elimina sesiones inactivas para liberar memoria"""
    current_time = datetime.now()
    inactive_sessions = []
    
    for session_id, timestamp in list(last_activity.items()):
        if current_time - timestamp > SESSION_TIMEOUT:
            inactive_sessions.append(session_id)
    
    for session_id in inactive_sessions:
        if session_id in conversation_history:
            del conversation_history[session_id]
        if session_id in last_activity:
            del last_activity[session_id]
        if session_id in processing_queries:
            del processing_queries[session_id]
    
    if inactive_sessions:
        logger.info(f"Eliminadas {len(inactive_sessions)} sesiones inactivas")

def get_conversation_history(session_id):
    """Obtiene o crea un historial de conversación para el ID de sesión dado"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in conversation_history:
        conversation_history[session_id] = []
        logger.info(f"Creada nueva sesión con ID: {session_id}")
    
    # Actualizar timestamp de última actividad
    last_activity[session_id] = datetime.now()
    
    return session_id, conversation_history[session_id]

def add_to_history(session_id, role, content):
    """Añade un mensaje al historial de conversación"""
    if session_id in conversation_history:
        conversation_history[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        # Actualizar timestamp de última actividad
        last_activity[session_id] = datetime.now()
        logger.debug(f"Añadido mensaje de {role} a sesión {session_id}")
    
def format_conversation_history(history, max_messages=5):
    """Formatea el historial de conversación para incluirlo en el prompt"""
    if not history:
        return []
        
    # Limitar a los últimos N mensajes para no sobrecargar el contexto
    recent_history = history[-max_messages:] if len(history) > max_messages else history
    formatted_history = []
    
    for message in recent_history:
        formatted_history.append({
            "role": "user" if message['role'] == 'user' else "assistant",
            "content": message['content']
        })
    
    return formatted_history

# -------------------------- Configuración de la aplicación --------------------------
app = Flask(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY no configurada en variables de entorno.")
    sys.exit("Configuración del servidor incompleta. Contacte al administrador.")

# Configurar LiteLLM para Google AI Studio
litellm.set_verbose = True

# System prompt para información del CV - placeholder para contenido real del CV
system_prompt = """
Eres un asistente de IA especializado en representar y discutir la información sobre el CV (Curriculum Vitae) de Marcos.
Tu propósito es proporcionar respuestas precisas, útiles y detalladas sobre el historial profesional de esta persona.

INSTRUCCIONES:
1. Responde preguntas sobre el contenido del CV de manera profesional y amigable.
2. Si la pregunta es sobre detalles específicos del CV, proporciona una respuesta completa.
3. Si la pregunta es sobre consejos profesionales generales relacionados con los campos del CV, brinda orientación útil.
4. Mantén siempre un tono profesional.

CONTENIDO DEL CV:
# Marcos García Estévez

## Información de Contacto:
**LinkedIn**: https://www.linkedin.com/in/marcosgarest  
**Web Personal**: https://warcos.dev

---

## Resumen Profesional
Desarrollador especializado en IA y tecnologías emergentes, con experiencia en machine learning y sistemas basados en LLM. Con sólidas habilidades de trabajo en equipo y conocimientos de infraestructura que me permiten desarrollar soluciones completas e innovadoras.

---

## Experiencia Laboral

### Administrador de Sistemas Cloud & WordPress
- **Empresa**: Grupo Oro (Prácticas)  
- **Periodo**: Marzo 2024 - Junio 2024  
- **Responsabilidades**:
  - Experiencia en despliegue y mantenimiento de sitios WordPress
  - Configuración avanzada de servidores
  - Gestión de DNS/Cloudflare
  - Optimización SEO

### LLM Quality Specialist
- **Empresa**: Outlier  
- **Periodo**: Mayo 2023 - Julio 2024  
- **Responsabilidades**:
  - Experiencia en comparación y refinamiento de modelos de lenguaje
  - Análisis de calidad de datos de entrenamiento
  - Detección de discrepancias y redacción técnica especializada

---

## Proyectos

### MIDAS
- **Descripción**: Trabajo Final de Máster (TFM) que propone un sistema innovador para automatizar el desarrollo de modelos de machine learning mediante una arquitectura multiagente. MIDAS cubre todo el ciclo de desarrollo ML: desde la generación de datos y visualizaciones, hasta el entrenamiento, validación y despliegue de modelos, permitiendo a profesionales de diversos niveles crear soluciones ML de forma ágil y accesible.
- **Arquitectura**: Sistema compuesto por 8 módulos especializados, de los cuales desarrollé 5 componentes principales:
  - **Midas Plot**: Generador de visualizaciones basado en CrewAI
  - **Midas Touch**: Sistema de agentes expertos para limpieza, entrenamiento y optimización de modelos. Subes los datos y te devuelve el joblib de un modelo de machine learning para predecir la columna especificada.
  - **Midas Assistant**: Interfaz central basada en LiteLLM con un system prompt especializado
  - **Midas Architect**: Componente RAG que utiliza Supabase como base de datos vectorial para consultar documentación de frameworks (Pydantic AI, LlamaIndex, CrewAI y AG2), implementado con Gemini 2.0 Flash y documentación scrapeada con Crawl4AI
  - **Midas Help**: Sistema RAG con LlamaIndex, que selecciona LLMs según la complejidad de la consulta usando BERT para clasificación y reranking para optimizar la recuperación de contexto
- **Contribución**: Además de desarrollar estos 5 componentes, me encargué de la arquitectura general del sistema y su despliegue.
- **Repositorio**: [Ver en GitHub](https://github.com/warc0s/MIDAS)

### LLM StoryTeller
- **Descripción**: Aplicación web interactiva que utiliza Large Language Models (LLMs) para ayudar a los usuarios a crear historias cautivadoras sin esfuerzo. Presenta características como creación interactiva con asistencia de IA, múltiples géneros y temas, desarrollo de personajes, sugerencias de trama y exportación en varios formatos. La aplicación utiliza modelos de lenguaje pequeños para generar segmentos de historia contextuales basados en la entrada del usuario, con un marco Streamlit que proporciona una interfaz intuitiva.
- **URL**: [https://llm-storyteller.streamlit.app](https://llm-storyteller.streamlit.app)
- **Repositorio**: [GitHub](https://github.com/warc0s/llm-storyteller)

### ChatCV
- **Descripción**: Currículum interactivo mediante un chatbot, implementando dos enfoques: LLM+RAG con Llama 3.3 70b + BGE-M3, y sistema basado en Gemini Flash con contexto extendido.
- **URL**: [https://chatbot.warcos.dev/](https://chatbot.warcos.dev/)

### Gather-Tracker
- **Fecha**: Junio 2024  
- **Descripción**: Herramienta desarrollada en JavaScript que extrae datos de "Gather Town" y los notifica a un canal de Telegram.  
- **Repositorio**: [GitHub](https://github.com/warc0s/Gather-Tracker)

### Fox-Detector
- **Fecha**: Enero 2024  
- **Descripción**: Modelo de visión por computadora basado en DenseNet121 para detectar zorros en imágenes, usando Keras y TensorFlow.  
- **Repositorio**: [GitHub](https://github.com/warc0s/Fox-Detector)

### XLSX a JSONL para Fine-Tuning de ChatGPT
- **Fecha**: Octubre 2024  
- **Descripción**: Script en Python que convierte archivos Excel (.xlsx) a formato JSONL, facilitando la creación de datasets estructurados de entrenamiento y validación para el fine-tuning de ChatGPT.  
- **Repositorio**: [GitHub](https://github.com/warc0s/xlsx-to-jsonl/)

### HDD Failure ML
- **Fecha**: Noviembre 2024
- **Descripción**: Sistema predictivo de fallos en discos duros mediante ensemble de modelos Random Forest y XGBoost con tres configuraciones optimizadas: máxima precisión (99%), máximo recall (86%) y balance F1 (80%). Aplicable universalmente a cualquier fabricante mediante datos SMART.

---

## Stack Tecnológico

- Desarrollo de modelos de Machine Learning y Deep Learning (TensorFlow/Keras) con énfasis en redes neuronales.
- Implementación de sistemas NLP y LLM con prompt engineering avanzado y fine-tuning de modelos.
- Desarrollo en Python con dominio de bibliotecas científicas (NumPy, Pandas...) para implementar soluciones IA.
- Experiencia en sistemas de embeddings y bases de datos vectoriales para procesamiento semántico.
- Desarrollo con frameworks de agentes autónomos o RAG (CrewAI, LlamaIndex) para automatización IA.

---

## Habilidades
- Capacidad analítica y resolución de problemas, con enfoque en descomposición de tareas complejas.
- Autodidacta, adquiriendo conocimiento de manera independiente.

---

## Educación

### CPIFP Alan Turing - Accenture
- **Máster FP en Inteligencia Artificial y Big Data**  
- **Periodo**: Sept. 2024 - Jun. 2025 (en curso)  
- Modalidad dual de 600 horas en oficinas de Accenture, con enfoque en IA, machine learning, programación y Big Data.

### EducacionIT
- **BootCamp Linux & Cloud ("Carrera Linux")**  
- **Periodo**: Jun. 2024 - Ago. 2025 (en curso)  
- Incluye Linux, redes, Shell, bases de datos SQL, ciberseguridad, hosting, cloud computing y contenedores.

### MEDAC
- **Ciclo Formativo de Grado Superior en Desarrollo de Aplicaciones Web (DAW)**  
- **Periodo**: Sept. 2022 - Jun. 2024  
- Especialización en áreas prácticas de la informática: HTML, CSS, JavaScript, PHP, Flask, SQL, y más.

### Universidad de Málaga
- **Grado en Ingeniería de Software (SIN FINALIZAR)**  
- **Periodo**: Sept. 2019 - Jun. 2022 
- Cursé tres años, enfocándome en desarrollar habilidades técnicas aplicables en IA y desarrollo web, con la posibilidad de retomar en el futuro.

---

## Idiomas

- **Español**: Nativo
- **Inglés**: Avanzado
  - Cambridge B1
  - EF SET English Certificate 71/100 (C1 Proficient)

---

## Licencias y Certificaciones

- **PCEP™ – Certified Entry-Level Python Programmer**  
  - **Expedición**: Oct. 2024

- **EF SET English Certificate 71/100 (C1 Proficient)**  
  - **Expedición**: Abr. 2024

- **B1 Preliminary English Test (Cambridge)**  
  - **Expedición**: Jun. 2017, ID de la credencial: 0058332818

Al responder preguntas:
- Sé conciso pero completo
- Destaca experiencias y habilidades relevantes
- Presenta la información de manera bien estructurada
- No inventes información que no esté presente en el CV
- Si no estás seguro sobre detalles específicos, indícalo en lugar de hacer suposiciones
- Si te preguntan sobre proyectos técnicos, proporciona detalles técnicos relevantes 
- Si te preguntan por habilidades, explica cómo se han aplicado en experiencias concretas

Esta información es de un profesional real, así que tus respuestas deben ser precisas y respetuosas con su carrera.
"""

# Sistema de concurrencia por sesión
processing_queries = {}  # {session_id: is_processing}
processing_lock = threading.Lock()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        selected_llm = data.get('llm', 'Automatico')
        session_id = data.get('session_id', '')
        
        if not user_input:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400

        # Limpieza periódica de sesiones inactivas (5% de probabilidad en cada petición)
        if random.random() < 0.05:
            clean_inactive_sessions()

        # Obtener o crear historial de conversación
        session_id, history = get_conversation_history(session_id)
        
        # Añadir la consulta actual al historial
        add_to_history(session_id, 'user', user_input)
        logger.info(f"Consulta registrada. Sesión {session_id} tiene {len(history)} mensajes")

        # Verificar si esta sesión específica ya está procesando una consulta
        with processing_lock:
            if session_id in processing_queries and processing_queries[session_id]:
                return jsonify({
                    'error': 'Ya estamos procesando tu consulta anterior. Por favor, espera un momento.'
                }), 429
            processing_queries[session_id] = True

        try:
            # Selección de LLM basada en la entrada del usuario
            if selected_llm != 'Automatico':
                if selected_llm == "Gemini 2.0 Flash":
                    model = "gemini/gemini-2.0-flash"
                    llm_usado = 'Gemini 2.0 Flash'
                elif selected_llm == "Gemini 2.0 Flash-Lite":
                    model = "gemini/gemini-2.0-flash-lite"
                    llm_usado = 'Gemini 2.0 Flash-Lite'
                else:
                    # Liberar el bloqueo antes de retornar error
                    with processing_lock:
                        processing_queries[session_id] = False
                    return jsonify({'error': 'Opción de LLM no reconocida.'}), 400
                logger.info(f"LLM forzado: {llm_usado}")
            else:
                # Flujo automático: clasificar el prompt con BERT
                dificultad = clasificar_dificultad(user_input)
                logger.info(f"Prompt clasificado con dificultad: {dificultad}")
                
                if dificultad == 2:
                    response_text = "Lo siento, no puedo responder a eso ya que no está relacionado la información del CV de Marcos. Si crees que se trata de un error, por favor, reformula la pregunta."
                    # Añadir la respuesta al historial
                    add_to_history(session_id, 'assistant', response_text)
                    
                    # Liberar el bloqueo
                    with processing_lock:
                        processing_queries[session_id] = False
                        
                    response_data = {
                        'response': response_text,
                        'status': 'success',
                        'llm': 'Bloqueado - PromptAnalysis',
                        'session_id': session_id
                    }
                    return jsonify(response_data)
                
                if dificultad == 0:
                    model = "gemini/gemini-2.0-flash-lite"
                    llm_usado = 'Gemini 2.0 Flash-Lite'
                elif dificultad == 1:
                    model = "gemini/gemini-2.0-flash"
                    llm_usado = 'Gemini 2.0 Flash'
                else:
                    # Liberar el bloqueo antes de retornar error
                    with processing_lock:
                        processing_queries[session_id] = False
                    return jsonify({'error': 'Clasificación de pregunta desconocida.'}), 400

            logger.info(f"Procesando consulta con {llm_usado}. Sesión: {session_id}")
            
            # Obtener historial formateado para la API
            formatted_history = format_conversation_history(history)
            
            # Crear mensajes para la llamada a la API
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(formatted_history)
            
            # Llamar a LiteLLM con el proveedor Google AI Studio
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=0.3,
                api_key=GOOGLE_API_KEY,
            )
            
            # Extraer texto de respuesta
            response_text = response.choices[0].message.content
            
            # Añadir la respuesta al historial
            add_to_history(session_id, 'assistant', response_text)
            logger.info(f"Respuesta añadida. Ahora la sesión tiene {len(history)} mensajes")
            
            response_data = {
                'response': response_text,
                'status': 'success',
                'llm': llm_usado,
                'session_id': session_id
            }
            
            return jsonify(response_data)
            
        finally:
            # Asegurar que el bloqueo siempre se libere
            with processing_lock:
                if session_id in processing_queries:
                    processing_queries[session_id] = False
        
    except Exception as e:
        # En caso de excepción, asegurarse de liberar el bloqueo
        if 'session_id' in locals() and session_id:
            with processing_lock:
                if session_id in processing_queries:
                    processing_queries[session_id] = False
        logger.error(f"Error procesando consulta: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_conversation_history():
    try:
        data = request.json
        session_id = data.get('session_id', '')
        
        if session_id and session_id in conversation_history:
            conversation_history[session_id] = []
            logger.info(f"Historial borrado para sesión {session_id}")
            # Actualizar timestamp de última actividad
            last_activity[session_id] = datetime.now()
            return jsonify({
                'status': 'success', 
                'message': 'Historial de conversación borrado',
                'session_id': session_id
            })
        return jsonify({
            'status': 'success', 
            'message': 'No hay historial para borrar',
            'session_id': session_id if session_id else str(uuid.uuid4())
        })
    except Exception as e:
        logger.error(f"Error borrando historial: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
