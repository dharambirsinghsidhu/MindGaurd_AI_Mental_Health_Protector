import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Tuple, Optional
import warnings
from lightgbm import LGBMClassifier
from streamlit_webrtc import webrtc_streamer
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import av
import os

# --- Import from your new files ---
from utils import DatabaseManager, AlertSystem # IMPORTANT CHANGE
from chatbot import render_chatbot_interface # NEW IMPORT
from voice_analyzer import render_voice_analysis_interface # NEW IMPORT

warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
    
    def fit(self, X, y):
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ModelManager:
    """Manages all ML models and predictions"""

    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all pre-trained models"""
        try:
            # Text analysis models
            with open('models/ensemble_model.pkl', 'rb') as f:
                self.models['text_ensemble'] = joblib.load(f)
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                self.models['tfidf'] = joblib.load(f)
            with open('models/label_encoder.pkl', 'rb') as f:
                self.models['label_encoder'] = joblib.load(f)

            # Behavioral models
            with open('models/best_rf_stress_level.pkl', 'rb') as f:
                self.models['stress'] = joblib.load(f)
            with open('models/best_rf_depression_score.pkl', 'rb') as f:
                self.models['depression'] = joblib.load(f)
            with open('models/best_rf_anxiety_score.pkl', 'rb') as f:
                self.models['anxiety'] = joblib.load(f)
            st.success("✅ All models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Model file not found or failed to load: {e}")
            st.stop()

    def predict_text_sentiment(self, text: str):
        # Transform the input text
        text_vector = self.models['tfidf'].transform([text])
        
        # Get the probabilities for each class for the first (and only) sample
        # The [0] at the end gets the probabilities for our single text input
        probabilities = self.models['text_ensemble'].predict_proba(text_vector)[0]
        
        # The confidence is the highest probability
        confidence = float(max(probabilities))
        
        # Get the index of the highest probability to find the predicted class
        prediction_index = probabilities.argmax()
        
        # Get the sentiment label using the index
        sentiment = self.models['label_encoder'].classes_[prediction_index]
        
        # Assess the risk level with the correct decimal confidence
        risk_level = self._assess_risk_level(sentiment, confidence)
        
        explanation = {
            'sentiment': sentiment,
            'confidence': confidence, # This will be a decimal like 0.6604
            'risk_level': risk_level
        }
        
        return sentiment, confidence, explanation

    def _assess_risk_level(self, sentiment: str, confidence: float) -> str:
        # This function is logically correct and does not need changes.
        sentiment_lower = sentiment.lower() if isinstance(sentiment, str) else ""

        if sentiment_lower == 'suicidal':
            return 'High'
        
        high_risk = ['anxiety', 'depression', 'stress']
        moderate_risk = ['anxiety', 'depression', 'stress', 'personality disorder', 'bipolar']

        if sentiment_lower in high_risk and confidence >= 0.8:
            return 'High'
        elif sentiment_lower in moderate_risk and confidence >= 0.5:
            return 'Moderate' # With confidence=0.6604, this will now be triggered
        else:
            return 'Low'


    def predict_behavioral_metrics(self, features: Dict):
        """Predict mental health scores from behavioral data (strictly using trained models)"""
        # Ensure the features are ordered as during training
        feature_array = np.array(list(features.values())).reshape(1, -1)
        stress = self.models['stress'].predict(feature_array)[0]
        depression = self.models['depression'].predict(feature_array)
        anxiety = self.models['anxiety'].predict(feature_array)
        results = {
            'stress_level': max(0, min(10, stress)),
            'depression_score': max(0, min(27, depression)),
            'anxiety_score': max(0, min(21, anxiety))
        }
        return results


class GamificationSystem:
    """Manages user gamification features"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tasks = [
            {"id": 1, "name": "Daily Check-in", "points": 10, "description": "Complete your daily mood assessment"},
            {"id": 2, "name": "Mindful Moment", "points": 15, "description": "Take 5 minutes for mindfulness"},
            {"id": 3, "name": "Physical Activity", "points": 20, "description": "Engage in 30 minutes of physical activity"},
            {"id": 4, "name": "Social Connection", "points": 25, "description": "Reach out to a friend or family member"},
            {"id": 5, "name": "Gratitude Practice", "points": 15, "description": "Write down 3 things you're grateful for"}
        ]
        
        self.badges = {
            "First Steps": {"requirement": 1, "description": "Complete your first assessment"},
            "Consistent": {"requirement": 7, "description": "7-day streak"},
            "Dedicated": {"requirement": 30, "description": "30-day streak"},
            "Mental Health Champion": {"requirement": 500, "description": "Earn 500 points"},
            "Wellness Warrior": {"requirement": 1000, "description": "Earn 1000 points"}
        }
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user gamification statistics"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT points, streak_days, badges FROM gamification WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            points, streak, badges_json = result
            badges = json.loads(badges_json) if badges_json else []
            return {
                'points': points or 0,
                'streak_days': streak or 0,
                'badges': badges
            }
        return {'points': 0, 'streak_days': 0, 'badges': []}
    
    def complete_task(self, user_id: int, task_id: int):
        """Complete a gamification task"""
        task = next((t for t in self.tasks if t['id'] == task_id), None)
        if task:
            # Award points
            self.db_manager.update_gamification(user_id, task['points'])
            
            # Check for new badges
            stats = self.get_user_stats(user_id)
            new_badges = self._check_badges(stats)
            
            for badge in new_badges:
                self.db_manager.update_gamification(user_id, 0, badge)
            
            return task['points'], new_badges
        return 0, []
    
    def _check_badges(self, stats: Dict) -> List[str]:
        """Check for earned badges"""
        new_badges = []
        current_badges = stats['badges']
        
        for badge_name, badge_info in self.badges.items():
            if badge_name not in current_badges:
                if badge_name == "First Steps" and stats['points'] > 0:
                    new_badges.append(badge_name)
                elif badge_name == "Consistent" and stats['streak_days'] >= 7:
                    new_badges.append(badge_name)
                elif badge_name == "Dedicated" and stats['streak_days'] >= 30:
                    new_badges.append(badge_name)
                elif badge_name == "Mental Health Champion" and stats['points'] >= 500:
                    new_badges.append(badge_name)
                elif badge_name == "Wellness Warrior" and stats['points'] >= 1000:
                    new_badges.append(badge_name)
        
        return new_badges

class VisualizationManager:
    """Handles all data visualizations and charts"""
    
    @staticmethod
    def create_mood_trend_chart(data: List[Dict]) -> go.Figure:
        """Create mood trend visualization"""
        if not data:
            return go.Figure()
        
        dates = [datetime.now() - timedelta(days=i) for i in range(len(data)-1, -1, -1)]
        scores = [d.get('overall_score', 5) for d in data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=scores,
            mode='lines+markers',
            name='Mood Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Mood Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Mood Score",
            hovermode='x'
        )
        
        return fig
    
    @staticmethod
    def create_assessment_radar_chart(scores: Dict) -> go.Figure:
        """Create radar chart for assessment scores"""
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) + 2]
                )),
            showlegend=True,
            title="Mental Health Assessment Overview"
        )
        
        return fig
    
    @staticmethod
    def create_risk_gauge(risk_level: str) -> go.Figure:
        """Create risk level gauge"""
        risk_values = {'Low': 1, 'Moderate': 2, 'High': 3}
        risk_colors = {'Low': 'green', 'Moderate': 'yellow', 'High': 'red'}
        
        value = risk_values.get(risk_level, 1)
        color = risk_colors.get(risk_level, 'green')
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level"},
            gauge = {
                'axis': {'range': [None, 3]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "gray"},
                    {'range': [2, 3], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.5
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig

# Initialize managers
@st.cache_resource
def init_managers():
    """Initialize all manager classes"""
    db_manager = DatabaseManager() # This now comes from utils.py
    model_manager = ModelManager()
    alert_system = AlertSystem() # This now comes from utils.py
    gamification = GamificationSystem(db_manager)
    viz_manager = VisualizationManager()
    
    return db_manager, model_manager, alert_system, gamification, viz_manager

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="MindGuard AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .task-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize managers
    db_manager, model_manager, alert_system, gamification, viz_manager = init_managers()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 MindGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Mental Health Monitoring & Intervention System</p>', unsafe_allow_html=True)
    
    # Sidebar for user management
    with st.sidebar:
        st.header("👤 User Profile")

        # User login/registration
        username = st.text_input("Username")
        if username:
            user_id = db_manager.get_user_id(username)
            if not user_id:
                st.info("New user! Let's set up your profile.")
                email = st.text_input("Your Email")
                emergency_contact = st.text_input("Emergency Contact Email")

                # Basic email validation can be added here if desired

                if st.button("Create Profile"):
                    user_id = db_manager.create_user(username, email, emergency_contact)
                    st.success("Profile created successfully!")
                    if st.button("Continue"):
                        st.experimental_rerun()
            else:
                st.success(f"Welcome back, {username}!")
                stats = gamification.get_user_stats(user_id)
                st.subheader("🎮 Your Progress")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Points", stats['points'])
                with col2:
                    st.metric("Streak", f"{stats['streak_days']} days")
                if stats['badges']:
                    st.write("🏆 **Badges Earned:**")
                    for badge in stats['badges']:
                        st.write(f"• {badge}")
    
    # Main content
    if username and db_manager.get_user_id(username):
        user_id = db_manager.get_user_id(username)

        tabs_list = [
            "💬 Daily Check-in", 
            "📊 Behavioral Assessment", 
            "🔬 Clinical Assessments",
            "🎯 Wellness Tasks",
            "📈 Dashboard",
            "🤖 AI Chatbot", 
            "🎙️ Voice Analysis" 
        ]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs_list)
        
        with tab1:
            st.header("Daily Mental Health Check-in")
            st.write("How are you feeling today? Share your thoughts and let our AI analyze your mental state.")
            
            user_input = st.text_area(
                "Tell us about your current state of mind:",
                placeholder="I'm feeling a bit stressed today because of work deadlines...",
                height=150
            )
            
            if st.button("Analyze My Response", type="primary"):
                if user_input:
                    with st.spinner("Analyzing your response..."):
                        sentiment, confidence, explanation = model_manager.predict_text_sentiment(user_input)

                    # Convert sentiment if needed
                    sentiment = sentiment.item() if hasattr(sentiment, 'item') else sentiment

                    # Convert confidence if needed
                    import numpy as np

                    if isinstance(confidence, (np.ndarray, list)):
                        confidence = float(max(confidence))
                    else:
                        confidence = float(confidence)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col3:
                        st.metric("Risk Level", explanation['risk_level'])

                    risk_gauge = viz_manager.create_risk_gauge(explanation['risk_level'])
                    st.plotly_chart(risk_gauge, use_container_width=True)

                    alert_needed, alert_reasons = alert_system.check_alert_conditions(explanation)  # Use explanation here
                    
                    if alert_needed:
                        user_email, emergency_email = db_manager.get_user_emails(user_id)
                        
                        # Create user_info dictionary for the alert message content
                        user_info = {
                            'username': username,
                            'email': user_email,
                            'emergency_contact': emergency_email
                        }

                        # IMPORTANT: Define the recipient list to ONLY include the emergency contact
                        recipients = []
                        if emergency_email:
                            recipients.append(emergency_email)
                        else:
                            st.warning("No emergency contact email found to send an alert.")

                        # Call send_alert with the specific recipients list
                        if recipients:
                            alert_system.send_alert(user_info, alert_reasons, explanation, recipients)

                    def convert_ndarrays(obj):
                        if isinstance(obj, dict):
                            return {k: convert_ndarrays(v) for k,v in obj.items()}
                        elif isinstance(obj, (np.ndarray, list, tuple)):
                            return [convert_ndarrays(i) for i in obj]
                        elif isinstance(obj, (np.generic,)):
                            return obj.item()
                        else:
                            return obj

                    clean_explanation = convert_ndarrays(explanation)

                    db_manager.save_assessment(user_id, 'text_analysis', clean_explanation, explanation['risk_level'])
                    points, new_badges = gamification.complete_task(user_id, 1)

                    st.success(f"✅ Assessment completed! You earned {points} points!")
                    if new_badges:
                        st.balloons()
                        st.success(f"🏆 New badge(s) earned: {', '.join(new_badges)}")
                else:
                    st.warning("Please enter your thoughts before analyzing.")
        
        with tab2:
            st.header("Behavioral Metrics Assessment")
            st.write("Please provide information about your lifestyle and current situation for comprehensive analysis.")
            
            with st.form("behavioral_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    age = st.slider("Age", 18, 80, 30)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
                    employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired"])
                    sleep_hours = st.slider("Sleep Hours per Night", 3, 12, 7)
                with col2:
                    activity_hours = st.slider("Physical Activity Hours per Week", 0, 20, 5)
                    social_support = st.slider("Social Support Score (1-10)", 1, 10, 6)
                    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
                    chronic_illness = st.selectbox("Chronic Illnesses", ["Yes", "No"])
                    medication = st.selectbox("Currently on Medication", ["Yes", "No"])
                with col3:
                    therapy = st.selectbox("Receiving Therapy", ["Yes", "No"])
                    meditation = st.selectbox("Practice Meditation", ["Yes", "No"])
                    substance_use = st.selectbox("Substance Use", ["None", "Occasional", "Regular"])
                    financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 5)
                    work_stress = st.slider("Work Stress (1-10)", 1, 10, 5)
                    self_esteem = st.slider("Self-Esteem Score (1-10)", 1, 10, 6)
                    life_satisfaction = st.slider("Life Satisfaction (1-10)", 1, 10, 6)
                    loneliness = st.slider("Loneliness Score (1-10)", 1, 10, 4)
                
                submitted = st.form_submit_button("Analyze Behavioral Metrics", type="primary")
                
                if submitted:
                    features = {
                        'Age': age,
                        'Gender': 1 if gender == 'Male' else (2 if gender == 'Female' else 3),
                        'Education_Level': {'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4, 'Other': 2}.get(education, 2),
                        'Employment_Status': {'Employed': 1, 'Unemployed': 2, 'Student': 3, 'Retired': 4}.get(employment, 1),
                        'Sleep_Hours': sleep_hours,
                        'Physical_Activity_Hrs': activity_hours,
                        'Social_Support_Score': social_support,
                        'Family_History_Mental_Illness': 1 if family_history == 'Yes' else 0,
                        'Chronic_Illnesses': 1 if chronic_illness == 'Yes' else 0,
                        'Medication_Use': 1 if medication == 'Yes' else 0,
                        'Therapy': 1 if therapy == 'Yes' else 0,
                        'Meditation': 1 if meditation == 'Yes' else 0,
                        'Substance_Use': {'None': 0, 'Occasional': 1, 'Regular': 2}.get(substance_use, 0),
                        'Financial_Stress': financial_stress,
                        'Work_Stress': work_stress,
                        'Self_Esteem_Score': self_esteem,
                        'Life_Satisfaction_Score': life_satisfaction,
                        'Loneliness_Score': loneliness
                    }

                    with st.spinner("Analyzing behavioral patterns..."):
                        results = model_manager.predict_behavioral_metrics(features)

                    st.subheader("🎯 Assessment Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        stress_level = results['stress_level']
                        stress_level = stress_level.item() if hasattr(stress_level, 'item') else float(stress_level)
                        st.metric("Stress Level", f"{stress_level:.1f}/10",
                                delta=f"{'⚠️ High' if stress_level > alert_system.alert_thresholds['stress_level'] else '✅ Normal'}")
                    with col2:
                        depression_score = results['depression_score']
                        depression_score = depression_score.item() if hasattr(depression_score, 'item') else float(depression_score)
                        st.metric("Depression Score", f"{depression_score:.1f}/27",
                                delta=f"{'⚠️ High' if depression_score > alert_system.alert_thresholds['depression_score'] else '✅ Normal'}")
                    with col3:
                        anxiety_score = results['anxiety_score']
                        anxiety_score = anxiety_score.item() if hasattr(anxiety_score, 'item') else float(anxiety_score)
                        st.metric("Anxiety Score", f"{anxiety_score:.1f}/21",
                                delta=f"{'⚠️ High' if anxiety_score > alert_system.alert_thresholds['anxiety_score'] else '✅ Normal'}")

                    radar_chart = viz_manager.create_assessment_radar_chart(results)
                    st.plotly_chart(radar_chart, use_container_width=True)

                    user_email, emergency_email = db_manager.get_user_emails(user_id)
                    user_info = {
                        'username': username,
                        'email': user_email,
                        'emergency_contact': emergency_email
                    }

                    alert_needed, alert_reasons = alert_system.check_alert_conditions(results)
                    if alert_needed:
                        # IMPORTANT: Define the recipient list to ONLY include the emergency contact
                        recipients = []
                        if emergency_email:
                            recipients.append(emergency_email)
                        else:
                            st.warning("No emergency contact email found to send an alert.")

                        # Call send_alert with the specific recipients list
                        if recipients:
                            alert_system.send_alert(user_info, alert_reasons, results, recipients)

                    # Determine overall risk level string for saving
                    risk_level = 'High' if any(
                        results[k] > alert_system.alert_thresholds.get(k, float('inf'))
                        for k in results if k in alert_system.alert_thresholds
                    ) else 'Low'

                    # Clean numpy arrays/scalars for JSON serialization before saving
                    import numpy as np

                    def convert_ndarrays(obj):
                        if isinstance(obj, dict):
                            return {k: convert_ndarrays(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_ndarrays(i) for i in obj]
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.generic):  # numpy scalar
                            return obj.item()
                        else:
                            return obj

                    clean_results = convert_ndarrays(results)
                    db_manager.save_assessment(user_id, 'behavioral', clean_results, risk_level)

                    
                    points, new_badges = gamification.complete_task(user_id, 2)
                    st.success(f"✅ Behavioral assessment completed! You earned {points} points!")
                    if new_badges:
                        st.balloons()
                        st.success(f"🏆 New badge(s) earned: {', '.join(new_badges)}")

        
        with tab3:
            st.header("Clinical Mental Health Assessments")
            st.write("Complete standardized clinical assessments for comprehensive mental health evaluation.")
            
            assessment_type = st.selectbox("Choose Assessment", ["DASS-21", "Beck Depression Inventory (BDI)", "Beck Anxiety Inventory (BAI)"])
            
            if assessment_type == "DASS-21":
                st.subheader("DASS-21 (Depression, Anxiety & Stress Scale)")
                st.write("Rate how much each statement applied to you over the past week.")
                
                questions = [
                    "I found it hard to wind down",
                    "I was aware of dryness of my mouth",
                    "I couldn't seem to experience any positive feeling at all",
                    "I experienced breathing difficulty",
                    "I found it difficult to work up the initiative to do things",
                    "I tended to over-react to situations",
                    "I experienced trembling",
                    "I felt that I was using a lot of nervous energy",
                    "I was worried about situations in which I might panic",
                    "I felt that I had nothing to look forward to",
                    "I found myself getting agitated",
                    "I found it difficult to relax",
                    "I felt down-hearted and blue",
                    "I was intolerant of anything that kept me from getting on with what I was doing",
                    "I felt I was close to panic",
                    "I was unable to become enthusiastic about anything",
                    "I felt I wasn't worth much as a person",
                    "I felt that I was rather touchy",
                    "I was aware of the action of my heart in the absence of physical exertion",
                    "I felt scared without any good reason",
                    "I felt that life was meaningless"
                ]
                
                scores = []
                with st.form("dass21_form"):
                    for i, question in enumerate(questions):
                        score = st.radio(
                            f"Q{i+1}: {question}",
                            options=[0, 1, 2, 3],
                            format_func=lambda x: ["Never", "Sometimes", "Often", "Almost Always"][x],
                            horizontal=True,
                            key=f"dass_{i}"
                        )
                        scores.append(score)
                    
                    if st.form_submit_button("Calculate DASS-21 Score", type="primary"):
                        # Calculate subscale scores
                        depression = sum(scores[i] for i in [2, 4, 9, 12, 15, 16, 20]) * 2
                        anxiety = sum(scores[i] for i in [1, 3, 6, 8, 14, 18, 19]) * 2
                        stress = sum(scores[i] for i in [0, 5, 7, 10, 11, 13, 17]) * 2
                        
                        results = {
                            'depression_score': depression,
                            'anxiety_score': anxiety,
                            'stress_score': stress
                        }
                        
                        # Display results with interpretations
                        st.subheader("📊 DASS-21 Results")
                        col1, col2, col3 = st.columns(3)
                        
                        def interpret_dass_depression(score):
                            if score < 10: return "Normal"
                            elif score < 14: return "Mild"
                            elif score < 21: return "Moderate"
                            elif score < 28: return "Severe"
                            else: return "Extremely Severe"
                        
                        def interpret_dass_anxiety(score):
                            if score < 8: return "Normal"
                            elif score < 10: return "Mild"
                            elif score < 15: return "Moderate"
                            elif score < 20: return "Severe"
                            else: return "Extremely Severe"
                        
                        def interpret_dass_stress(score):
                            if score < 15: return "Normal"
                            elif score < 19: return "Mild"
                            elif score < 26: return "Moderate"
                            elif score < 34: return "Severe"
                            else: return "Extremely Severe"
                        
                        with col1:
                            dep_interp = interpret_dass_depression(depression)
                            st.metric("Depression", f"{depression}/42", delta=dep_interp)
                        
                        with col2:
                            anx_interp = interpret_dass_anxiety(anxiety)
                            st.metric("Anxiety", f"{anxiety}/42", delta=anx_interp)
                        
                        with col3:
                            stress_interp = interpret_dass_stress(stress)
                            st.metric("Stress", f"{stress}/42", delta=stress_interp)
                        
                        # Check for high-risk scores
                        high_risk = depression >= 28 or anxiety >= 20 or stress >= 34
                        if high_risk:
                            alert_reasons = []
                            if depression >= 28: alert_reasons.append("Extremely Severe Depression")
                            if anxiety >= 20: alert_reasons.append("Extremely Severe Anxiety")
                            if stress >= 34: alert_reasons.append("Extremely Severe Stress")
                            
                            user_email, emergency_email = db_manager.get_user_emails(user_id)

                            user_info = {
                                'username': username,
                                'email': user_email,
                                'emergency_contact': emergency_email
                            }

                            alert_system.send_alert(user_info, alert_reasons, results)
                        
                        # Save and award points
                        risk_level = 'High' if high_risk else ('Moderate' if max(results.values()) > 15 else 'Low')
                        db_manager.save_assessment(user_id, 'DASS-21', results, risk_level)
                        points, new_badges = gamification.complete_task(user_id, 1)
                        st.success(f"✅ DASS-21 assessment completed! You earned {points} points!")
            
            elif assessment_type == "Beck Depression Inventory (BDI)":
                st.subheader("Beck Depression Inventory (BDI-II)")
                st.write("Choose the statement that best describes how you have been feeling during the past two weeks.")
                
                # Simplified BDI questions for demo
                bdi_questions = [
                    {"question": "Sadness", "options": ["I do not feel sad", "I feel sad much of the time", "I am sad all the time", "I am so sad that I can't stand it"]},
                    {"question": "Pessimism", "options": ["I am not discouraged about my future", "I feel more discouraged about my future than usual", "I do not expect things to work out for me", "I feel my future is hopeless"]},
                    {"question": "Past failure", "options": ["I do not feel like a failure", "I have failed more than I should have", "As I look back, I see failures", "I feel I am a total failure as a person"]},
                    {"question": "Loss of pleasure", "options": ["I get as much pleasure as I ever did", "I don't enjoy things as much as I used to", "I get very little pleasure from things", "I can't get any pleasure from things"]},
                    {"question": "Guilt feelings", "options": ["I don't feel particularly guilty", "I feel guilty much of the time", "I feel quite guilty most of the time", "I feel guilty all of the time"]},
                ]
                
                with st.form("bdi_form"):
                    bdi_scores = []
                    for i, item in enumerate(bdi_questions):
                        score = st.radio(
                            f"{item['question']}:",
                            options=list(range(len(item['options']))),
                            format_func=lambda x, opts=item['options']: opts[x],
                            key=f"bdi_{i}"
                        )
                        bdi_scores.append(score)
                    
                    if st.form_submit_button("Calculate BDI Score", type="primary"):
                        total_score = sum(bdi_scores)
                        
                        def interpret_bdi(score):
                            if score < 14: return "Minimal depression"
                            elif score < 20: return "Mild depression"
                            elif score < 29: return "Moderate depression"
                            else: return "Severe depression"
                        
                        interpretation = interpret_bdi(total_score)
                        
                        st.subheader("📊 BDI Results")
                        st.metric("Depression Score", f"{total_score}/63", delta=interpretation)
                        
                        # Progress bar
                        st.progress(total_score / 63)
                        
                        results = {'depression_score': total_score, 'interpretation': interpretation}
                        
                        # Check for alerts
                        if total_score >= 29:
                            user_email, emergency_email = db_manager.get_user_emails(user_id)

                            user_info = {
                                'username': username,
                                'email': user_email,
                                'emergency_contact': emergency_email
                            }

                            alert_system.send_alert(user_info, ["Severe Depression (BDI)"], results)
                        
                        # Save and award points
                        risk_level = 'High' if total_score >= 29 else ('Moderate' if total_score >= 20 else 'Low')
                        db_manager.save_assessment(user_id, 'BDI', results, risk_level)
                        points, new_badges = gamification.complete_task(user_id, 1)
                        st.success(f"✅ BDI assessment completed! You earned {points} points!")
            
            else:  # BAI
                st.subheader("Beck Anxiety Inventory (BAI)")
                st.write("Rate how much you have been bothered by each symptom during the past month.")
                
                bai_symptoms = [
                    "Numbness or tingling",
                    "Feeling hot",
                    "Wobbliness in legs",
                    "Unable to relax",
                    "Fear of worst happening",
                    "Dizzy or lightheaded",
                    "Heart pounding/racing",
                    "Unsteady",
                    "Terrified or afraid",
                    "Nervous"
                ]
                
                with st.form("bai_form"):
                    bai_scores = []
                    for i, symptom in enumerate(bai_symptoms):
                        score = st.radio(
                            f"{symptom}:",
                            options=[0, 1, 2, 3],
                            format_func=lambda x: ["Not at all", "Mildly", "Moderately", "Severely"][x],
                            horizontal=True,
                            key=f"bai_{i}"
                        )
                        bai_scores.append(score)
                    
                    if st.form_submit_button("Calculate BAI Score", type="primary"):
                        total_score = sum(bai_scores)
                        
                        def interpret_bai(score):
                            if score < 8: return "Minimal anxiety"
                            elif score < 16: return "Mild anxiety"
                            elif score < 26: return "Moderate anxiety"
                            else: return "Severe anxiety"
                        
                        interpretation = interpret_bai(total_score)
                        
                        st.subheader("📊 BAI Results")
                        st.metric("Anxiety Score", f"{total_score}/63", delta=interpretation)
                        
                        # Progress bar
                        st.progress(total_score / 63)
                        
                        results = {'anxiety_score': total_score, 'interpretation': interpretation}
                        
                        # Check for alerts
                        if total_score >= 26:
                            user_email, emergency_email = db_manager.get_user_emails(user_id)

                            user_info = {
                                'username': username,
                                'email': user_email,
                                'emergency_contact': emergency_email
                            }

                            alert_system.send_alert(user_info, ["Severe Anxiety (BAI)"], results)
                        
                        # Save and award points
                        risk_level = 'High' if total_score >= 26 else ('Moderate' if total_score >= 16 else 'Low')
                        db_manager.save_assessment(user_id, 'BAI', results, risk_level)
                        points, new_badges = gamification.complete_task(user_id, 1)
                        st.success(f"✅ BAI assessment completed! You earned {points} points!")
        
        with tab4:
            st.header("🎯 Daily Wellness Tasks")
            st.write("Complete these evidence-based mental health activities to improve your wellbeing and earn points!")
            
            # Display available tasks
            for task in gamification.tasks:
                with st.container():
                    st.markdown(f'<div class="task-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{task['name']}** ({task['points']} points)")
                        st.write(task['description'])
                    
                    with col2:
                        if st.button(f"Complete", key=f"task_{task['id']}", type="secondary"):
                            points, new_badges = gamification.complete_task(user_id, task['id'])
                            st.success(f"✅ Task completed! +{points} points")
                            if new_badges:
                                st.balloons()
                                st.success(f"🏆 New badge(s): {', '.join(new_badges)}")
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Wellness tips
            st.subheader("💡 Mental Health Tips")
            tips = [
                "Practice deep breathing for 2-3 minutes when feeling stressed",
                "Take a 10-minute walk outdoors to boost mood and energy",
                "Write down 3 things you're grateful for each day",
                "Connect with a friend or family member regularly",
                "Maintain a consistent sleep schedule",
                "Limit caffeine and alcohol intake",
                "Practice mindfulness or meditation daily",
                "Engage in physical activity you enjoy",
                "Set small, achievable daily goals",
                "Seek professional help when needed"
            ]
            
            tip_of_day = tips[datetime.now().day % len(tips)]
            st.info(f"💡 **Tip of the Day:** {tip_of_day}")

        with tab5:  
            st.header("📈 Mental Health Dashboard")
            st.write("Track your mental health journey and progress over time.")

            # User statistics overview
            stats = gamification.get_user_stats(user_id)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Points", stats['points'], delta="🎯")
            with col2:
                st.metric("Current Streak", f"{stats['streak_days']} days", delta="🔥")
            with col3:
                st.metric("Badges Earned", len(stats['badges']), delta="🏆")
            with col4:
                # Total assessments count from DB
                conn = sqlite3.connect(db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM assessments WHERE user_id = ?', (user_id,))
                total_assessments = cursor.fetchone()[0]
                conn.close()
                st.metric("Assessments", f"{total_assessments}", delta="📊")

            # Retrieve assessments for last 7 days, mapping scores over dates
            assessment_map = db_manager.get_user_assessments(user_id, days_back=7)
            dates = [(datetime.now() - timedelta(days=i)).date() for i in reversed(range(7))]
            
            last_known_score = None
            daily_scores = []
            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in assessment_map:
                    last_known_score = assessment_map[date_str]
                # Use 'stress_level' or fallback default 5 if no data
                overall_score = last_known_score.get('stress_level', 5) if last_known_score else 5
                daily_scores.append({'date': date, 'overall_score': overall_score})

            # Create mood trend chart from real data
            mood_chart = viz_manager.create_mood_trend_chart(daily_scores)
            st.plotly_chart(mood_chart, use_container_width=True)

            # Recent assessments summary table, latest 5 assessments
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, assessment_type, risk_level
                FROM assessments
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 5
            ''', (user_id,))
            recent_rows = cursor.fetchall()
            conn.close()

            recent_data = {
                'Date': [row[0][:10] for row in recent_rows],
                'Type': [row[1] for row in recent_rows],
                'Risk Level': [row[2] for row in recent_rows],
                'Status': ['✅ Complete'] * len(recent_rows)
            }
            recent_df = pd.DataFrame(recent_data)
            st.subheader("📋 Recent Assessment Summary")
            st.dataframe(recent_df, use_container_width=True)

            # Display earned badges
            if stats['badges']:
                st.subheader("🏆 Your Badges")
                badge_cols = st.columns(min(len(stats['badges']), 4))
                for i, badge in enumerate(stats['badges']):
                    with badge_cols[i % 4]:
                        st.success(f"🏆 {badge}")

            # Mental Health Resources Info
            st.subheader("🆘 Mental Health Resources")
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **Emergency Hotlines:**
                - National Suicide Prevention Lifeline: 988
                - Crisis Text Line: Text HOME to 741741
                - National Domestic Violence Hotline: 1-800-799-7233
                """)
            with col2:
                st.info("""
                **Professional Help:**
                - Find a therapist: Psychology Today
                - Mental health apps: Headspace, Calm, BetterHelp
                - Support groups: NAMI, Mental Health America
                """)

        with tab6:
            st.header("Advanced AI Chatbot")
            # This function is imported from your chatbot.py file
            render_chatbot_interface(db_manager, user_id, username)

        with tab7:
            st.header("In-Depth Voice Analysis")
            # This function is imported from your voice_analyzer.py file
            render_voice_analysis_interface(db_manager, user_id, username)


if __name__ == "__main__":
    main()