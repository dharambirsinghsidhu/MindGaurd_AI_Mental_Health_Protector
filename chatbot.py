import streamlit as st
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from datetime import datetime
import re
import time
from typing import Dict, List, Tuple, Optional
import sqlite3

class MentalHealthChatbot:
    """Advanced mental health chatbot with emotion analysis and therapeutic responses"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.conversation_history = []
        self.load_models()
        self.init_therapeutic_responses()
    
    def load_models(self):
        """Load lightweight Hugging Face models for mental health analysis"""
        try:
            # Emotion analysis model (lightweight)
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Use CPU to avoid GPU requirements
            )
            
            # Mental health classification model (alternative lightweight option)
            self.mental_health_classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=-1
            )
            
            # Sentiment analysis for backup
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            st.success("Chatbot models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading chatbot models: {e}")
            # Fallback to basic models
            self.emotion_analyzer = pipeline("sentiment-analysis", device=-1)
            self.mental_health_classifier = None
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
    
    def init_therapeutic_responses(self):
        """Initialize therapeutic response templates"""
        self.response_templates = {
            'depression': [
                "I hear that you're going through a difficult time. Depression can feel overwhelming, but you're not alone in this.",
                "It takes courage to express these feelings. Have you considered talking to a mental health professional?",
                "Your feelings are valid. Small steps toward self-care can make a difference. What's one small thing you could do for yourself today?"
            ],
            'anxiety': [
                "Anxiety can be really challenging to deal with. What you're experiencing is real and valid.",
                "It sounds like you're feeling anxious. Sometimes grounding techniques can help - try focusing on 5 things you can see around you.",
                "Anxiety often makes things feel worse than they are. Have you tried any breathing exercises or mindfulness practices?"
            ],
            'stress': [
                "Stress is something many people struggle with. You're taking a positive step by talking about it.",
                "It sounds like you're under a lot of pressure. What are some ways you typically cope with stress?",
                "Managing stress is important for your wellbeing. Have you been able to take any breaks or do something you enjoy recently?"
            ],
            'anger': [
                "I can sense you're feeling frustrated or angry. These emotions are normal, though they can be difficult to handle.",
                "Anger often signals that something important to you feels threatened. What might be underneath these feelings?",
                "It's okay to feel angry. What healthy ways have you found to express or work through these feelings?"
            ],
            'sadness': [
                "I'm sorry you're feeling sad. It's important to acknowledge and honor these feelings.",
                "Sadness is a natural human emotion. Is there anything specific that's been weighing on your mind?",
                "Your sadness matters. Sometimes talking about what's troubling us can provide some relief."
            ],
            'fear': [
                "Fear can be paralyzing, but you're brave for acknowledging it. What's been causing you to feel afraid?",
                "It's natural to feel scared sometimes. Fear often protects us, but it can also hold us back.",
                "Facing our fears takes courage. What's one small step you could take to address what's worrying you?"
            ],
            'positive': [
                "It's wonderful to hear that you're feeling well! What's been contributing to your positive mood?",
                "I'm glad you're having a good day. It's important to recognize and celebrate these moments.",
                "Your positive energy is great to see. What activities or thoughts help you maintain this mindset?"
            ],
            'suicidal': [
                "I'm very concerned about what you've shared. Your life has value and there are people who want to help.",
                "These thoughts must be incredibly painful. Please reach out to a crisis helpline: 988 (Suicide & Crisis Lifeline).",
                "You don't have to go through this alone. Emergency help is available 24/7 at 988 or text HOME to 741741."
            ],
            'general': [
                "Thank you for sharing that with me. How are you feeling about the situation?",
                "I'm here to listen and support you. What's been on your mind lately?",
                "It's important to check in with yourself. How would you describe your overall mood today?"
            ]
        }
        
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living', 'better off dead',
            'hurt myself', 'self harm', 'cutting', 'overdose', 'want to die'
        ]
        
        self.wellness_suggestions = {
            'depression': [
                "Try to maintain a daily routine, even a simple one",
                "Consider gentle exercise like a short walk",
                "Reach out to a trusted friend or family member",
                "Practice self-compassion - treat yourself as you would a good friend"
            ],
            'anxiety': [
                "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8",
                "Practice grounding: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste",
                "Consider writing down your worries to externalize them",
                "Limit caffeine if it tends to increase your anxiety"
            ],
            'stress': [
                "Take regular breaks, even just 5-10 minutes",
                "Try progressive muscle relaxation",
                "Consider time management techniques like the Pomodoro method",
                "Make sure you're getting adequate sleep"
            ]
        }
    
    def analyze_message(self, message: str) -> Dict:
        """Analyze user message for emotions and mental health indicators"""
        # Check for crisis keywords first
        message_lower = message.lower()
        has_crisis_keywords = any(keyword in message_lower for keyword in self.crisis_keywords)
        
        # Emotion analysis
        try:
            emotions = self.emotion_analyzer(message)
            primary_emotion = emotions[0]['label'].lower() if emotions else 'neutral'
            emotion_confidence = emotions[0]['score'] if emotions else 0.5
        except:
            primary_emotion = 'neutral'
            emotion_confidence = 0.5
        
        # Sentiment analysis
        try:
            sentiment = self.sentiment_analyzer(message)
            sentiment_label = sentiment[0]['label'].lower()
            sentiment_score = sentiment[0]['score']
        except:
            sentiment_label = 'neutral'
            sentiment_score = 0.5
        
        # Risk assessment
        risk_level = self.assess_risk_level(primary_emotion, sentiment_label, has_crisis_keywords, message)
        
        return {
            'emotion': primary_emotion,
            'emotion_confidence': emotion_confidence,
            'sentiment': sentiment_label,
            'sentiment_score': sentiment_score,
            'risk_level': risk_level,
            'has_crisis_keywords': has_crisis_keywords,
            'timestamp': datetime.now().isoformat()
        }
    
    def assess_risk_level(self, emotion: str, sentiment: str, has_crisis: bool, message: str) -> str:
        """Assess mental health risk level based on analysis"""
        if has_crisis:
            return 'Critical'
        
        high_risk_emotions = ['sadness', 'fear', 'anger']
        moderate_risk_emotions = ['disgust', 'surprise']
        
        # Check message length and content depth
        message_indicators = self.check_message_indicators(message)
        
        if emotion in high_risk_emotions and sentiment in ['negative', 'label_0']:
            if message_indicators['severity'] >= 3:
                return 'High'
            else:
                return 'Moderate'
        elif emotion in moderate_risk_emotions:
            return 'Moderate'
        elif emotion == 'joy' and sentiment in ['positive', 'label_2']:
            return 'Low'
        else:
            return 'Moderate'
    
    def check_message_indicators(self, message: str) -> Dict:
        """Analyze message content for additional risk indicators"""
        message_lower = message.lower()
        
        # Severity indicators
        severe_words = ['hopeless', 'worthless', 'trapped', 'burden', 'empty', 'numb']
        moderate_words = ['tired', 'sad', 'worried', 'stressed', 'overwhelmed']
        
        severity_score = 0
        severity_score += sum(2 for word in severe_words if word in message_lower)
        severity_score += sum(1 for word in moderate_words if word in message_lower)
        
        return {
            'severity': min(severity_score, 5),
            'word_count': len(message.split()),
            'has_severity_indicators': severity_score > 0
        }
    
    def generate_response(self, message: str, analysis: Dict) -> str:
        """Generate appropriate therapeutic response"""
        emotion = analysis['emotion']
        risk_level = analysis['risk_level']
        
        # Crisis response
        if analysis['has_crisis_keywords'] or risk_level == 'Critical':
            return self.get_crisis_response()
        
        # Map emotions to response categories
        emotion_mapping = {
            'sadness': 'depression',
            'fear': 'anxiety',
            'anger': 'anger',
            'joy': 'positive',
            'surprise': 'general',
            'disgust': 'general'
        }
        
        response_category = emotion_mapping.get(emotion, 'general')
        
        # Get base response
        responses = self.response_templates.get(response_category, self.response_templates['general'])
        base_response = np.random.choice(responses)
        
        # Add wellness suggestions for negative emotions
        if response_category in self.wellness_suggestions and risk_level in ['High', 'Moderate']:
            suggestions = self.wellness_suggestions[response_category]
            suggestion = np.random.choice(suggestions)
            base_response += f"\n\nHere's something that might help: {suggestion}"
        
        # Add professional help recommendation for high risk
        if risk_level == 'High':
            base_response += "\n\nI encourage you to consider speaking with a mental health professional who can provide personalized support."
        
        return base_response
    
    def get_crisis_response(self) -> str:
        """Generate crisis intervention response"""
        return """🚨 I'm very concerned about what you've shared. Your life has value and you don't have to go through this alone.

**Immediate Help Available:**
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911

Please reach out to one of these resources right away. There are people who want to help you through this difficult time.

If you're in immediate danger, please call 911 or go to your nearest emergency room."""
    
    def save_conversation(self, user_id: int, message: str, response: str, analysis: Dict):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Create conversations table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_message TEXT,
                bot_response TEXT,
                analysis_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute('''
            INSERT INTO conversations (user_id, user_message, bot_response, analysis_data)
            VALUES (?, ?, ?, ?)
        ''', (user_id, message, response, json.dumps(analysis)))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Retrieve recent conversation history"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, bot_response, analysis_data, timestamp
            FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'user_message': row[0],
                'bot_response': row[1],
                'analysis': json.loads(row[2]),
                'timestamp': row[3]
            })
        
        return list(reversed(history))  # Show oldest first

def render_chatbot_interface(db_manager, user_id: int, username: str):
    """Render the chatbot interface"""
    st.header("🤖 Mental Health AI Chatbot")
    st.write("I'm here to listen and provide support. Share what's on your mind, and I'll respond with care and understanding.")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MentalHealthChatbot(db_manager)
    
    chatbot = st.session_state.chatbot
    
    # Initialize chat history in session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = chatbot.get_conversation_history(user_id, 20)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            # User message
            with st.chat_message("user"):
                st.write(msg['user_message'])
                st.caption(f"Emotion: {msg['analysis']['emotion']} | Risk: {msg['analysis']['risk_level']}")
            
            # Bot response
            with st.chat_message("assistant"):
                st.write(msg['bot_response'])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to session state immediately
        st.session_state.chat_messages.append({
            'user_message': user_input,
            'bot_response': "Analyzing...",
            'analysis': {'emotion': 'processing', 'risk_level': 'unknown'},
            'timestamp': datetime.now().isoformat()
        })
        
        # Analyze message
        with st.spinner("Understanding your message..."):
            analysis = chatbot.analyze_message(user_input)
            response = chatbot.generate_response(user_input, analysis)
        
        # Update the last message with actual response
        st.session_state.chat_messages[-1] = {
            'user_message': user_input,
            'bot_response': response,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to database
        chatbot.save_conversation(user_id, user_input, response, analysis)
        
        # Check for alerts
        if analysis['risk_level'] in ['High', 'Critical']:
            st.error("🚨 High risk detected - Alert system activated")
            
            # Get user contact info
            user_email, emergency_email = db_manager.get_user_emails(user_id)
            
            # Import AlertSystem from main app
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
                    [f"Chatbot detected {analysis['risk_level']} risk"], 
                    analysis, 
                    recipients
                )
        
        # Rerun to show updated chat
        st.rerun()
    
    # Sidebar with chat statistics
    with st.sidebar:
        st.subheader("💬 Chat Statistics")
        
        if st.session_state.chat_messages:
            emotions = [msg['analysis'].get('emotion', 'unknown') for msg in st.session_state.chat_messages]
            emotion_counts = pd.Series(emotions).value_counts()
            
            st.write("**Emotion Distribution:**")
            for emotion, count in emotion_counts.head(5).items():
                st.write(f"• {emotion.title()}: {count}")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            st.rerun()
        
        if st.button("Export Chat"):
            chat_data = pd.DataFrame(st.session_state.chat_messages)
            csv = chat_data.to_csv(index=False)
            st.download_button(
                label="Download Chat History",
                data=csv,
                file_name=f"chat_history_{username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )