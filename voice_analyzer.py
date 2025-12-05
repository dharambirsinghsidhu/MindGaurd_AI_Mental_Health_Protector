import streamlit as st
import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import librosa
import io
import tempfile
import os
from datetime import datetime
import json
import sqlite3
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
try:
    import soundfile as sf
    import scipy.signal
except ImportError:
    st.error("Audio processing libraries not found. Install soundfile and scipy.")


def convert_np_types(d):
    """Convert numpy data types in dict to native Python types for JSON serialization."""
    if isinstance(d, dict):
        return {k: convert_np_types(v) for k, v in d.items()}
    elif isinstance(d, (np.float32, np.float64)):
        return float(d)
    elif isinstance(d, (np.int32, np.int64)):
        return int(d)
    elif isinstance(d, list):
        return [convert_np_types(i) for i in d]
    else:
        return d


class VoiceAnalyzer:
    """Voice analysis for mental health assessment using audio features and speech-to-text"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.load_models()
        self.init_audio_features()
    
    def load_models(self):
        """Load Hugging Face models for voice analysis"""
        try:
            # Speech-to-text model (lightweight)
            self.speech_to_text = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",  # Small model for quick processing
                device=-1
            )
            
            # Emotion analysis from text
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1
            )
            
            # Voice emotion classification (if available)
            try:
                self.voice_emotion_analyzer = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    device=-1
                )
            except:
                self.voice_emotion_analyzer = None
                st.info("Voice emotion model not available, using text analysis only")
            
            st.success("Voice analysis models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading voice models: {e}")
            self.speech_to_text = None
            self.emotion_analyzer = None
            self.voice_emotion_analyzer = None
    
    def init_audio_features(self):
        """Initialize audio feature extraction parameters"""
        self.sample_rate = 16000  # Standard for most models
        self.frame_length = 2048
        self.hop_length = 512
    
    def extract_audio_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract acoustic features from audio for mental health analysis"""
        try:
            # Resample if necessary
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # Basic acoustic features
            features = {}
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # 2. Energy and loudness
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            features['rms_energy_mean'] = np.mean(rms_energy)
            features['rms_energy_std'] = np.std(rms_energy)
            
            # 3. Zero crossing rate (speech rhythm)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 4. Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # 5. Speech rate (approximate)
            features['speech_duration'] = len(audio_data) / self.sample_rate
            
            # 6. Silence analysis
            silence_threshold = np.percentile(rms_energy, 20)
            silence_frames = np.sum(rms_energy < silence_threshold)
            features['silence_ratio'] = silence_frames / len(rms_energy)
            
            return features
            
        except Exception as e:
            st.error(f"Error extracting audio features: {e}")
            return {}
    
    def analyze_voice_patterns(self, features: Dict) -> Dict:
        """Analyze voice patterns for mental health indicators"""
        mental_health_indicators = {}
        
        # Depression indicators
        # Low energy, monotone speech, slow speech rate
        depression_score = 0
        if features.get('rms_energy_mean', 0) < 0.02:  # Low energy
            depression_score += 1
        if features.get('pitch_range', 0) < 50:  # Monotone
            depression_score += 1
        if features.get('silence_ratio', 0) > 0.3:  # Long pauses
            depression_score += 1
        
        mental_health_indicators['depression_likelihood'] = min(depression_score / 3, 1.0)
        
        # Anxiety indicators
        # High pitch variability, fast speech, tension
        anxiety_score = 0
        if features.get('pitch_std', 0) > 30:  # High pitch variability
            anxiety_score += 1
        if features.get('zcr_mean', 0) > 0.1:  # Higher zero crossing rate
            anxiety_score += 1
        if features.get('spectral_centroid_std', 0) > 500:  # Spectral variability
            anxiety_score += 1
        
        mental_health_indicators['anxiety_likelihood'] = min(anxiety_score / 3, 1.0)
        
        # Stress indicators
        # Combination of anxiety and depression indicators
        stress_score = (mental_health_indicators['depression_likelihood'] + 
                       mental_health_indicators['anxiety_likelihood']) / 2
        mental_health_indicators['stress_likelihood'] = stress_score
        
        # Overall risk assessment
        max_likelihood = max(mental_health_indicators.values())
        if max_likelihood > 0.7:
            mental_health_indicators['risk_level'] = 'High'
        elif max_likelihood > 0.4:
            mental_health_indicators['risk_level'] = 'Moderate'
        else:
            mental_health_indicators['risk_level'] = 'Low'
        
        return mental_health_indicators
    
    def process_audio_file(self, audio_file) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
        """Process uploaded audio file for analysis"""
        try:
            # Read audio file
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            
            # Extract acoustic features
            acoustic_features = self.extract_audio_features(audio_data, sample_rate)
            
            # Analyze voice patterns
            voice_analysis = self.analyze_voice_patterns(acoustic_features)
            
            # Convert to text
            transcribed_text = None
            if self.speech_to_text:
                try:
                    # Save temporary file for speech-to-text
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        sf.write(tmp_file.name, audio_data, sample_rate)
                        result = self.speech_to_text(tmp_file.name)
                        transcribed_text = result['text'] if isinstance(result, dict) else str(result)
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    
                except Exception as e:
                    st.warning(f"Speech-to-text failed: {e}")
                    transcribed_text = None
            
            return transcribed_text, acoustic_features, voice_analysis
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None, None, None
    
    def save_voice_analysis(self, user_id: int, transcribed_text: str, acoustic_features: Dict, voice_analysis: Dict):
        """Save voice analysis results to database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Create voice_analysis table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                transcribed_text TEXT,
                acoustic_features TEXT,
                voice_analysis TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        clean_acoustic_features = convert_np_types(acoustic_features)
        clean_voice_analysis = convert_np_types(voice_analysis)

        cursor.execute('''
            INSERT INTO voice_analysis (user_id, transcribed_text, acoustic_features, voice_analysis)
            VALUES (?, ?, ?, ?)
        ''', (
            user_id, 
            transcribed_text or "", 
            json.dumps(clean_acoustic_features), 
            json.dumps(clean_voice_analysis)
        ))

        
        conn.commit()
        conn.close()

def render_voice_analysis_interface(db_manager, user_id: int, username: str):
    """Render the voice analysis interface"""
    st.header("🎙️ Voice Mental Health Analysis")
    st.write("Upload an audio recording or speak directly to analyze your voice patterns for mental health insights.")
    
    # Initialize voice analyzer
    if 'voice_analyzer' not in st.session_state:
        st.session_state.voice_analyzer = VoiceAnalyzer(db_manager)
    
    voice_analyzer = st.session_state.voice_analyzer
    
    # Create tabs for different input methods
    upload_tab, record_tab, results_tab = st.tabs(["📁 Upload Audio", "🎤 Record Audio", "📊 Analysis Results"])
    
    with upload_tab:
        st.subheader("Upload Audio File")
        st.write("Upload an audio file (WAV, MP3, M4A) for analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Recommended: 30 seconds to 2 minutes of clear speech"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("Analyze Uploaded Audio", type="primary"):
                with st.spinner("Processing audio file..."):
                    transcribed_text, acoustic_features, voice_analysis = voice_analyzer.process_audio_file(uploaded_file)
                
                if transcribed_text or voice_analysis:
                    # Store results in session state
                    st.session_state.voice_results = {
                        'transcribed_text': transcribed_text,
                        'acoustic_features': acoustic_features,
                        'voice_analysis': voice_analysis,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save to database
                    voice_analyzer.save_voice_analysis(user_id, transcribed_text, acoustic_features, voice_analysis)
                    
                    st.success("Audio analysis completed! Check the Results tab.")
                    
                    # Check for alerts
                    if voice_analysis and voice_analysis.get('risk_level') == 'High':
                        user_email, emergency_email = db_manager.get_user_emails(user_id)
                        
                        from app import AlertSystem
                        alert_system = AlertSystem()
                        
                        user_info = {
                            'username': username,
                            'email': user_email,
                            'emergency_contact': emergency_email
                        }
                        
                        recipients = [emergency_email] if emergency_email else []
                        
                        if recipients:
                            alert_system.send_alert(
                                user_info,
                                ["High risk detected in voice analysis"],
                                voice_analysis,
                                recipients
                            )
                else:
                    st.error("Failed to process audio file. Please try again with a different file.")
    
    with record_tab:
        st.subheader("Record Your Voice")
        st.write("Click the button below to record your voice directly in the browser.")
        
        # Note: streamlit-webrtc is complex for voice recording
        # For simplicity, we'll provide instructions for file upload
        st.info("""
        🎤 **Recording Instructions:**
        
        To record your voice:
        1. Use your device's voice recorder app
        2. Speak naturally for 30 seconds to 2 minutes
        3. Save the recording as a WAV or MP3 file
        4. Upload it using the "Upload Audio" tab
        
        **What to say:**
        - Describe how you're feeling today
        - Talk about recent events or experiences
        - Read a paragraph from a book or article
        - Share what's been on your mind lately
        """)
        
        # Alternative: Simple browser-based recording (requires user permission)
        if st.button("🎤 Start Browser Recording"):
            st.info("Browser recording would require additional JavaScript integration. Please use the upload method for now.")
    
    with results_tab:
        st.subheader("📊 Voice Analysis Results")
        
        if 'voice_results' in st.session_state:
            results = st.session_state.voice_results
            
            # Display transcribed text
            if results['transcribed_text']:
                st.subheader("📝 Transcribed Text")
                st.text_area("What you said:", results['transcribed_text'], height=100, disabled=True)
                
                # Analyze transcribed text for emotions
                if voice_analyzer.emotion_analyzer:
                    try:
                        text_emotions = voice_analyzer.emotion_analyzer(results['transcribed_text'])
                        primary_emotion = text_emotions[0]['label']
                        emotion_confidence = text_emotions[0]['score']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Detected Emotion", primary_emotion.title())
                        with col2:
                            st.metric("Confidence", f"{emotion_confidence:.2%}")
                    except Exception as e:
                        st.warning(f"Text emotion analysis failed: {e}")
            
            # Display voice analysis results
            if results['voice_analysis']:
                st.subheader("🎵 Voice Pattern Analysis")
                
                voice_analysis = results['voice_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    depression_pct = voice_analysis.get('depression_likelihood', 0) * 100
                    st.metric("Depression Indicators", f"{depression_pct:.1f}%")
                
                with col2:
                    anxiety_pct = voice_analysis.get('anxiety_likelihood', 0) * 100
                    st.metric("Anxiety Indicators", f"{anxiety_pct:.1f}%")
                
                with col3:
                    stress_pct = voice_analysis.get('stress_likelihood', 0) * 100
                    st.metric("Stress Indicators", f"{stress_pct:.1f}%")
                
                # Risk level display
                risk_level = voice_analysis.get('risk_level', 'Low')
                risk_colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
                st.markdown(f"**Overall Risk Level:** :{risk_colors[risk_level]}[{risk_level}]")
            
            # Display acoustic features
            if results['acoustic_features']:
                with st.expander("🔊 Detailed Acoustic Analysis"):
                    features = results['acoustic_features']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Speech Characteristics:**")
                        st.write(f"• Duration: {features.get('speech_duration', 0):.1f} seconds")
                        st.write(f"• Average Pitch: {features.get('pitch_mean', 0):.1f} Hz")
                        st.write(f"• Pitch Variation: {features.get('pitch_std', 0):.1f}")
                    
                    with col2:
                        st.write("**Voice Quality:**")
                        st.write(f"• Energy Level: {features.get('rms_energy_mean', 0):.3f}")
                        st.write(f"• Speech Rate: {features.get('zcr_mean', 0):.3f}")
                        st.write(f"• Silence Ratio: {features.get('silence_ratio', 0):.2%}")
            
            # Recommendations based on analysis
            st.subheader("💡 Personalized Recommendations")
            if results['voice_analysis']:
                voice_analysis = results['voice_analysis']
                
                if voice_analysis.get('depression_likelihood', 0) > 0.5:
                    st.info("Your voice patterns suggest low energy. Consider gentle physical activity and social connection.")
                
                if voice_analysis.get('anxiety_likelihood', 0) > 0.5:
                    st.info("Your voice shows signs of tension. Deep breathing exercises and mindfulness might help.")
                
                if voice_analysis.get('stress_likelihood', 0) > 0.5:
                    st.info("Stress indicators detected. Consider stress management techniques and adequate rest.")
                
                if voice_analysis.get('risk_level') == 'Low':
                    st.success("Your voice patterns suggest good mental health. Keep up the positive practices!")
        else:
            st.info("No voice analysis results available. Please upload or record an audio file in the other tabs.")
        
        # Historical voice analysis
        st.subheader("📈 Voice Analysis History")
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DATE(timestamp) as date, voice_analysis
            FROM voice_analysis
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            history_data = []
            for row in rows:
                try:
                    analysis = json.loads(row[1])
                    history_data.append({
                        'Date': row[0],
                        'Depression': f"{analysis.get('depression_likelihood', 0)*100:.1f}%",
                        'Anxiety': f"{analysis.get('anxiety_likelihood', 0)*100:.1f}%",
                        'Stress': f"{analysis.get('stress_likelihood', 0)*100:.1f}%",
                        'Risk Level': analysis.get('risk_level', 'Unknown')
                    })
                except:
                    continue
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No voice analysis history available yet.")