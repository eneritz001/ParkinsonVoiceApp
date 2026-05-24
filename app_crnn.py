import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import io
from streamlit_mic_recorder import mic_recorder
import auth_manager as auth

# Configuración de página estilo móvil
st.set_page_config(page_title="NeuroVoice AI", page_icon="🧠", layout="centered")

# Cargar modelo con caché para no saturar la RAM
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("parkinson_crnn_model.h5")

model = load_model()

# Inicializar estados de navegación
if 'user' not in st.session_state: st.session_state.user = None
if 'screen' not in st.session_state: st.session_state.screen = 'login'

# Función de preprocesamiento desde memoria (sin archivos temporales)
# def preprocess_audio_bytes(audio_bytes):
#     audio_file = io.BytesIO(audio_bytes)
#     y, sr = librosa.load(audio_file, sr=22050, duration=3)
#     if len(y) < sr * 3:
#         y = np.pad(y, (0, sr * 3 - len(y)))
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     return mfcc.T[np.newaxis, ..., np.newaxis]

# def preprocess_audio_bytes(audio_bytes):
#     try:
#         from pydub import AudioSegment
#         import io
        
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         wav_io = io.BytesIO()
#         audio_segment.export(wav_io, format="wav")
#         wav_io.seek(0)
        
#         y, sr = librosa.load(wav_io, sr=22050, duration=3)
#         if len(y) < sr * 3:
#             y = np.pad(y, (0, sr * 3 - len(y)))
            
#         # EXTRAER LOS 3 CANALES IGUAL QUE EN EL ENTRENAMIENTO
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         delta_mfcc = librosa.feature.delta(mfcc)
#         delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
#         features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
#         features = np.transpose(features, (1, 0, 2))
        
#         # Añadir la dimensión del batch -> (1, frames, 40, 3)
#         return features[np.newaxis, ...]
        
#     except Exception as e:
#         st.error(f"Error al procesar el audio: {e}")
#         return None


# def preprocess_audio_bytes(audio_bytes):
#     try:
#         from pydub import AudioSegment
#         import io
#         import librosa
#         import numpy as np
        
#         # 1. Convertir los bytes del navegador a WAV en memoria usando pydub
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         wav_io = io.BytesIO()
#         audio_segment.export(wav_io, format="wav")
#         wav_io.seek(0)
        
#         # 2. Cargar con librosa a la tasa de muestreo estándar (22050 Hz)
#         # Forzamos una duración máxima de 3 segundos
#         y, sr = librosa.load(wav_io, sr=22050, duration=3)
        
#         # 3. EMPALME TEMPORAL ESTRICTO (La clave para solucionar el error de dimensiones)
#         target_length = sr * 3  # Exactamente 66150 muestras para 3 segundos
#         if len(y) < target_length:
#             # Si es más corto, rellenamos con silencio (ceros) al final
#             y = np.pad(y, (0, target_length - len(y)))
#         elif len(y) > target_length:
#             # Si es más largo por milisegundos, lo recortamos exactamente ahí
#             y = y[:target_length]
            
#         # 4. Extraer los 3 canales idénticos al entrenamiento
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         delta_mfcc = librosa.feature.delta(mfcc)
#         delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
#         # Unimos los canales -> forma (40, frames, 3)
#         features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        
#         # Permutamos los ejes para que el tiempo sea el primero -> (frames, 40, 3)
#         features = np.transpose(features, (1, 0, 2))
        
#         # Añadimos la dimensión del lote (batch) requerida por Keras -> (1, frames, 40, 3)
#         return features[np.newaxis, ...]
        
#     except Exception as e:
#         st.error(f"Error crítico en el preprocesamiento de audio: {e}")
#         return None

def preprocess_audio_bytes(audio_bytes):
    try:
        import imageio_ffmpeg
        import subprocess
        import tempfile
        import os
        import librosa
        import numpy as np
        
        # 1. Obtener la ruta exacta del traductor de audio que instalamos
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # 2. Guardar los bytes del navegador en un archivo temporal seguro
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
            temp_in.write(audio_bytes)
            temp_in_path = temp_in.name
            
        temp_out_path = temp_in_path + ".wav"
        
        # 3. CONVERSIÓN DIRECTA SIN PYDUB (Evita el WinError 2)
        # Le ordenamos a ffmpeg que convierta el archivo de forma silenciosa
        subprocess.run([
            ffmpeg_exe, 
            "-y",             # Sobrescribir si ya existe
            "-i", temp_in_path, # Archivo de entrada (WebM del micro)
            temp_out_path     # Archivo de salida (WAV limpio)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 4. Cargar el audio convertido a la duración matemática exacta
        y, sr = librosa.load(temp_out_path, sr=22050, duration=3)
        
        # 5. Limpieza vital: borrar archivos temporales para no llenar tu disco duro
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        
        # 6. Empalme temporal (Truncar / Rellenar a 3 segundos exactos)
        target_length = sr * 3  
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        elif len(y) > target_length:
            y = y[:target_length]
            
        # 7. Extracción de las matrices 3D (MFCC + Deltas)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        features = np.transpose(features, (1, 0, 2))
        
        return features[np.newaxis, ...]
        
    except Exception as e:
        import streamlit as st
        st.error(f"Error crítico en el preprocesamiento de audio: {e}")
        return None




# --- PANTALLAS ---

if st.session_state.screen == 'login':
    st.title("🧬 NeuroVoice AI")
    tab1, tab2 = st.tabs(["Entrar", "Registrarse"])
    
    with tab1:
        email = st.text_input("Email", key="l_email")
        pw = st.text_input("Contraseña", type="password", key="l_pw")
        if st.button("Iniciar Sesión", use_container_width=True):
            user, msg = auth.login_user(email, pw)
            if user:
                st.session_state.user = user
                st.session_state.screen = 'home'
                st.experimental_rerun()()
            else: st.error(msg)
            
    with tab2:
        name = st.text_input("Nombre Completo")
        new_email = st.text_input("Email")
        new_pw = st.text_input("Contraseña", type="password")
        if st.button("Crear Cuenta", use_container_width=True):
            success, msg = auth.register_user(name, new_email, new_pw)
            if success: st.success(msg)
            else: st.error(msg)

elif st.session_state.screen == 'home':
    user = st.session_state.user
    st.title(f"👋 ¡Hola, {user['name']}!")
    
    # Gráfica de evolución
    if user['history']:
        st.subheader("Tu Evolución de Estabilidad")
        data = [h['average'] for h in user['history']]
        st.line_chart(data)
    else:
        st.info("Aún no tienes registros. ¡Realiza tu primer chequeo!")

    if st.button("🎤 Iniciar Nuevo Chequeo Dual", use_container_width=True, type="primary"):
        st.session_state.screen = 'test'
        st.experimental_rerun()
        
    if st.button("👤 Mi Perfil y Ajustes", use_container_width=True):
        st.session_state.screen = 'profile'
        st.experimental_rerun()

# elif st.session_state.screen == 'test':
#     st.title("🎤 Captura Dual Robusta")
#     st.write("Sigue las instrucciones para una mayor precisión.")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("**Test 1: Vocal A**")
#         audio1 = mic_recorder(start_prompt="Grabar 'A'", stop_prompt="Detener", key='v1')
#     with col2:
#         st.write("**Test 2: Lectura**")
#         audio2 = mic_recorder(start_prompt="Grabar Frase", stop_prompt="Detener", key='v2')

#     if audio1 and audio2:
#         if st.button("Analizar Ambos Audios", use_container_width=True):
#             with st.spinner("La IA está promediando los resultados..."):
#                 in1 = preprocess_audio_bytes(audio1['bytes'])
#                 in2 = preprocess_audio_bytes(audio2['bytes'])
#                 p1 = model.predict(in1)[0][0]
#                 p2 = model.predict(in2)[0][0]
                
#                 # Guardar y actualizar sesión
#                 new_data = auth.save_test_result(st.session_state.user['email'], p1, p2)
#                 st.session_state.user = new_data
#                 st.session_state.last_results = (p1, p2)
#                 st.session_state.screen = 'results'
#                 st.rerun()
    
#     if st.button("Cancelar"):
#         st.session_state.screen = 'home'
#         st.rerun()

elif st.session_state.screen == 'test':
    st.title("🎤 Captura Dual Robusta")
    st.write("Por favor, realiza ambas grabaciones antes de proceder al análisis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test 1: Vocal A")
        st.info("Mantén la vocal 'A' de forma sostenida durante 3 segundos.")
        audio1 = mic_recorder(start_prompt="🔴 Grabar 'A'", stop_prompt="⏹️ Detener Test 1", key='v1')
        if audio1 and 'bytes' in audio1:
            st.success("✅ Audio 1 recibido correctamente")
            
    with col2:
        st.subheader("Test 2: Lectura")
        st.info("Lee la frase en voz alta: 'El sol sale por el este'.")
        audio2 = mic_recorder(start_prompt="🔴 Grabar Frase", stop_prompt="⏹️ Detener Test 2", key='v2')
        if audio2 and 'bytes' in audio2:
            st.success("✅ Audio 2 recibido correctamente")

    st.markdown("---")

    # VALIDACIÓN CLAVE: El botón solo se activa si ambos diccionarios contienen datos válidos
    if audio1 and audio2 and ('bytes' in audio1) and ('bytes' in audio2):
        if st.button("📊 Analizar Ambos Audios y Generar Diagnóstico", use_container_width=True, type="primary"):
            with st.spinner("La red neuronal CRNN está promediando las características espectrales..."):
                
                # Preprocesar de forma segura
                in1 = preprocess_audio_bytes(audio1['bytes'])
                in2 = preprocess_audio_bytes(audio2['bytes'])
                
                # Verificar que el traductor de pydub/librosa no haya devuelto None por fallo de formato
                if in1 is not None and in2 is not None:
                    p1 = model.predict(in1)[0][0]
                    p2 = model.predict(in2)[0][0]
                    
                    # Guardar en base de datos local e historial del usuario
                    new_data = auth.save_test_result(st.session_state.user['email'], p1, p2)
                    st.session_state.user = new_data
                    st.session_state.last_results = (p1, p2)
                    st.session_state.screen = 'results'
                    st.experimental_rerun()
                else:
                    st.error("❌ Uno de los audios no pudo ser decodificado. Por favor, intenta grabar de nuevo en un entorno más silencioso.")
    else:
        # Botón desactivado visualmente si falta alguna toma
        st.warning("⚠️ Esperando a que completes ambas grabaciones para poder activar el diagnóstico dual.")
    
    if st.button("⬅️ Cancelar y Volver al Inicio", use_container_width=True):
        st.session_state.screen = 'home'
        st.experimental_rerun()

elif st.session_state.screen == 'results':
    p1, p2 = st.session_state.last_results
    avg = (p1 + p2) / 2
    st.title("📊 Resultado del Diagnóstico")
    
    st.metric("Estabilidad Promedio", f"{round((1-avg)*100)}%")
    
    if avg > 0.5:
        st.warning("Se han detectado patrones de voz compatibles con Parkinson. Se recomienda consulta médica.")
    else:
        st.success("Tu voz muestra patrones estables y saludables.")
        
    st.write(f"Vocal: {round((1-p1)*100)}% | Lectura: {round((1-p2)*100)}%")
    
    if st.button("Volver al Inicio", use_container_width=True):
        st.session_state.screen = 'home'
        st.experimental_rerun()

elif st.session_state.screen == 'profile':
    st.title("👤 Mi Perfil")
    user = st.session_state.user
    st.write(f"**Nombre:** {user['name']}")
    st.write(f"**Email:** {user['email']}")
    
    if st.button("Cerrar Sesión", type="secondary"):
        st.session_state.user = None
        st.session_state.screen = 'login'
        st.experimental_rerun()
        
    if st.button("Volver"):
        st.session_state.screen = 'home'
        sst.experimental_rerun()