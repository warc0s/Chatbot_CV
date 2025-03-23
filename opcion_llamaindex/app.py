import os
import sys
import threading
import logging
import random
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    ChatPromptTemplate
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar app Flask
app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()

# Obtener claves de API de forma segura
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY no está configurada en las variables de entorno.")
    sys.exit("Configuración del servidor incompleta. Contacta al administrador.")

# Obtener la clave API de DeepInfra
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logger.error("DEEPINFRA_API_KEY no está configurada en las variables de entorno.")
    sys.exit("Configuración del servidor incompleta. Contacta al administrador.")

# Directorio para almacenar datos de índice
INDEX_STORAGE_PATH = os.path.join(os.getcwd(), "index_storage")

# Configuración global de LlamaIndex
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.3
)

# Configuración de DeepInfra Embeddings
Settings.embed_model = DeepInfraEmbeddingModel(
    model_id="BAAI/bge-m3",
    api_token=DEEPINFRA_API_KEY,
    normalize=True,
    text_prefix="text: ",
    query_prefix="query: "
)

# Configuración de chunk size
Settings.chunk_size = 512

# Definición del prompt mejorado para el chatbot de CV
cv_prompt_str = (
    "Eres un asistente de IA especializado en representar y discutir la información sobre el CV (Curriculum Vitae) de Marcos. "
    "Tu propósito es proporcionar respuestas precisas, útiles y detalladas sobre el historial profesional de esta persona.\n\n"
    "INSTRUCCIONES:\n"
    "1. Responde preguntas sobre el contenido del CV de manera profesional y amigable.\n"
    "2. Si la pregunta es sobre detalles específicos del CV, proporciona una respuesta completa.\n"
    "3. Si la pregunta es sobre consejos profesionales generales relacionados con los campos del CV, brinda orientación útil.\n"
    "4. Mantén siempre un tono profesional.\n\n"
    "CONTENIDO DEL CV:\n"
    "----------\n"
    "{context_str}\n"
    "----------\n\n"
    "Al responder preguntas:\n"
    "- Sé conciso pero completo\n"
    "- Destaca experiencias y habilidades relevantes\n"
    "- Presenta la información de manera bien estructurada\n"
    "- No inventes información que no esté presente en el CV\n"
    "- Si no estás seguro sobre detalles específicos, indícalo en lugar de hacer suposiciones\n"
    "- Si te preguntan sobre proyectos técnicos, proporciona detalles técnicos relevantes\n"
    "- Si te preguntan por habilidades, explica cómo se han aplicado en experiencias concretas\n"
    "Esta información es de un profesional real, así que tus respuestas deben ser precisas y respetuosas con su carrera.\n\n"
    "Pregunta: {query_str}\n\n"
    "Respuesta:"
)

# Crear template de chat con el prompt mejorado
chat_cv_msgs = [
    (
        "system",
        "Eres un asistente especializado en el CV de Marcos Garcia Estevez. Utiliza solo la información proporcionada para responder preguntas sobre su experiencia profesional, habilidades y formación."
    ),
    ("user", cv_prompt_str),
]
cv_template = ChatPromptTemplate.from_messages(chat_cv_msgs)

# Gestión de historiales de conversación
conversation_history = {}  # {session_id: [lista de mensajes]}
last_activity = {}  # {session_id: timestamp}
SESSION_TIMEOUT = timedelta(hours=12)  # Sesiones inactivas por más de 12 horas se eliminarán
processing_queries = {}  # {session_id: is_processing}
processing_lock = threading.Lock()

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
        logger.info(f"Añadido mensaje de {role} a sesión {session_id}")
    
def format_conversation_history(history, max_messages=5):
    """Formatea el historial de conversación para incluirlo en el prompt"""
    if not history:
        return ""
        
    # Limitar a los últimos N mensajes para no sobrecargar el contexto
    recent_history = history[-max_messages:] if len(history) > max_messages else history
    formatted_history = ""
    
    for message in recent_history:
        role = "Usuario" if message['role'] == 'user' else "Asistente"
        formatted_history += f"{role}: {message['content']}\n\n"
    
    return formatted_history

def modify_query_with_history(history, query_str):
    """Modifica la consulta para incluir el historial de conversación relevante"""
    if not history:
        return query_str
    
    # Formatear el historial de conversación
    formatted_history = format_conversation_history(history)
    
    # Construir la consulta mejorada que incluye el historial
    enhanced_query = (
        f"HISTORIAL DE CONVERSACIÓN RELEVANTE:\n"
        f"{formatted_history}\n"
        f"CONSULTA ACTUAL: {query_str}\n\n"
        f"Responde a la CONSULTA ACTUAL teniendo en cuenta el HISTORIAL DE CONVERSACIÓN RELEVANTE. "
        f"Si la consulta hace referencia a elementos mencionados anteriormente en la conversación, "
        f"asegúrate de contextualizar tu respuesta correctamente."
    )
    
    return enhanced_query

# Verificar la existencia del archivo info.md
execution_dir = os.getcwd()
md_filename = "info.md"
md_path = os.path.join(execution_dir, md_filename)

if not os.path.exists(md_path):
    logger.error(f"{md_filename} no encontrado en el directorio de ejecución: {execution_dir}")
    sys.exit("Archivo de información no encontrado. Contacta al administrador.")

# Función para crear o cargar índice
def get_or_create_index():
    # Intentar cargar el índice desde almacenamiento
    if os.path.exists(INDEX_STORAGE_PATH):
        try:
            logger.info("Cargando índice desde almacenamiento...")
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
            index = load_index_from_storage(storage_context=storage_context)
            logger.info("Índice cargado correctamente.")
            return index
        except Exception as e:
            logger.warning(f"Error al cargar el índice: {e}. Creando nuevo índice...")
    
    # Si no existe o hay error, crear nuevo índice
    logger.info("Creando nuevo índice vectorial...")
    documents = SimpleDirectoryReader(input_files=[md_path]).load_data()
    
    if not documents:
        logger.error(f"No se encontraron documentos con el nombre {md_filename}.")
        sys.exit("No se encontraron documentos válidos. Contacta al administrador.")
    
    # Crear el índice
    index = VectorStoreIndex.from_documents(documents)
    
    # Persistir el índice
    index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
    logger.info("Índice creado y persistido correctamente.")
    return index

# Obtener o crear índice
index = get_or_create_index()

# Crear motor de consultas con el prompt personalizado
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",
    text_qa_template=cv_template
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', '')
        
        if not user_input:
            return jsonify({'error': 'La consulta no puede estar vacía'}), 400

        # Limpieza periódica de sesiones inactivas (5% de probabilidad en cada petición)
        if random.random() < 0.05:
            clean_inactive_sessions()

        # Obtener o crear historial de conversación
        session_id, history = get_conversation_history(session_id)
        
        # Verificar si esta sesión específica ya está procesando una consulta
        with processing_lock:
            if session_id in processing_queries and processing_queries[session_id]:
                return jsonify({
                    'error': 'Ya estamos procesando tu consulta anterior. Por favor, espera un momento.'
                }), 429
            processing_queries[session_id] = True
        
        try:
            # Añadir la consulta actual al historial
            add_to_history(session_id, 'user', user_input)
            logger.info(f"Consulta registrada. Sesión {session_id} tiene {len(history)} mensajes")
            
            # Obtener historial previo (excluyendo la consulta actual recién añadida)
            previous_history = history[:-1] if len(history) > 1 else []
            
            # Modificar la consulta para incluir el historial relevante
            enhanced_query = modify_query_with_history(previous_history, user_input)
            logger.info(f"Consulta mejorada creada con contexto. Longitud historia: {len(previous_history)}")
            
            # Realizar la consulta con el contexto mejorado
            response = query_engine.query(enhanced_query)
            response_text = str(response)
            
            # Añadir la respuesta al historial
            add_to_history(session_id, 'assistant', response_text)
            logger.info(f"Respuesta añadida. Ahora la sesión tiene {len(history)} mensajes")
            
            response_data = {
                'response': response_text,
                'status': 'success',
                'llm': 'Llama 3.3 70B',
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
        logger.error(f"Error procesando consulta: {e}")
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
        logger.error(f"Error borrando historial: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)