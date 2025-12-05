import sqlite3
import json
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from typing import Dict, List, Tuple, Optional

class DatabaseManager:
    """Manages SQLite database operations for user data and history"""
    
    def __init__(self, db_path: str = "mindguard.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize ALL database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User profiles table (no change)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL, email TEXT,
                emergency_contact TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Assessment history table (no change)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                assessment_type TEXT, scores TEXT, risk_level TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Gamification table (no change)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gamification (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
                points INTEGER DEFAULT 0, streak_days INTEGER DEFAULT 0,
                last_activity DATE, badges TEXT DEFAULT '[]',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # --- NEW: Add table for Chatbot ---
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

        # --- NEW: Add table for Voice Analyzer ---
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
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str = "", emergency_contact: str = "") -> int:
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, email, emergency_contact) 
                VALUES (?, ?, ?)
            ''', (username, email, emergency_contact))
            user_id = cursor.lastrowid
            
            # Initialize gamification data
            cursor.execute('''
                INSERT INTO gamification (user_id) VALUES (?)
            ''', (user_id,))
            
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return self.get_user_id(username)
        finally:
            conn.close()
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user ID by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def get_user_emails(self, user_id: int) -> Tuple[Optional[str], Optional[str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT email, emergency_contact FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0], row[1]
        return None, None


    def save_assessment(self, user_id: int, assessment_type: str, scores: Dict, risk_level: str):
        """Save assessment results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO assessments (user_id, assessment_type, scores, risk_level)
            VALUES (?, ?, ?, ?)
        ''', (user_id, assessment_type, json.dumps(scores), risk_level))
        conn.commit()
        conn.close()
    
    def update_gamification(self, user_id: int, points: int, badge: str = None):
        """Update user gamification data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current data
        cursor.execute('SELECT points, badges, last_activity FROM gamification WHERE user_id = ?', (user_id,))
        current = cursor.fetchone()
        
        if current:
            current_points, badges_json, last_activity = current
            badges = json.loads(badges_json) if badges_json else []
            
            # Update points
            new_points = current_points + points
            
            # Add badge if provided
            if badge and badge not in badges:
                badges.append(badge)
            
            # Update streak
            today = datetime.now().date()
            if last_activity:
                last_date = datetime.strptime(last_activity, '%Y-%m-%d').date()
                if last_date == today - timedelta(days=1):
                    cursor.execute('UPDATE gamification SET streak_days = streak_days + 1 WHERE user_id = ?', (user_id,))
                elif last_date != today:
                    cursor.execute('UPDATE gamification SET streak_days = 1 WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('UPDATE gamification SET streak_days = 1 WHERE user_id = ?', (user_id,))
            
            cursor.execute('''
                UPDATE gamification 
                SET points = ?, badges = ?, last_activity = ?
                WHERE user_id = ?
            ''', (new_points, json.dumps(badges), today.strftime('%Y-%m-%d'), user_id))
        
        conn.commit()
        conn.close()

    def get_user_assessments(self, user_id: int, days_back: int = 7):
        """Fetch user assessment scores for last `days_back` days ordered by date ascending"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days_back - 1)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT DATE(timestamp) AS assessment_date, scores
            FROM assessments
            WHERE user_id = ? AND DATE(timestamp) >= ?
            ORDER BY timestamp ASC
        ''', (user_id, cutoff_str))

        rows = cursor.fetchall()
        conn.close()

        assessment_map = {}
        for date_str, scores_json in rows:
            scores = json.loads(scores_json)
            assessment_map[date_str] = scores
        
        return assessment_map
    

class AlertSystem:
    """Handles emergency alerts and notifications"""

    def __init__(self):
        self.alert_thresholds = {
            'depression_score': 20,
            'anxiety_score': 15,
            'stress_level': 8,
            'sentiment_risk': 'High'
        }
        # Replace with your actual sender email & app password (use secure secrets management in prod)
        self.from_email = "mailtestermailing@gmail.com"
        self.app_password = "cyzc ylns hoci qmno"

    def check_alert_conditions(self, results: dict) -> tuple:
        """Check if alert conditions are met"""
        alerts = []

        if 'depression_score' in results and results['depression_score'] >= self.alert_thresholds['depression_score']:
            alerts.append('High Depression Score')

        if 'anxiety_score' in results and results['anxiety_score'] >= self.alert_thresholds['anxiety_score']:
            alerts.append('High Anxiety Score')

        if 'stress_level' in results and results['stress_level'] >= self.alert_thresholds['stress_level']:
            alerts.append('High Stress Level')

        if 'risk_level' in results and results['risk_level'] == 'High':
            alerts.append('High Risk Sentiment')

        return len(alerts) > 0, alerts

    def send_email_alert(self, to_email: str, subject: str, body: str) -> bool:
        """Send email alert via Gmail SMTP with error handling and logging"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.from_email, self.app_password)
                server.sendmail(self.from_email, to_email, msg.as_string())

            st.success(f"Alert email successfully sent to {to_email}")
            return True
        except Exception as e:
            st.error(f"Failed to send alert email to {to_email}: {e}")
            return False


    def send_alert(self, user_info: dict, alert_reasons: list, results: dict, recipients: list) -> bool:
        """Send alert to the provided list of recipients and show alert in UI"""
        try:
            alert_message = self._create_alert_message(user_info['username'], alert_reasons, results)

            st.error("🚨 **ALERT TRIGGERED**")
            st.error(f"**Reasons:** {', '.join(alert_reasons)}")
            
            if not recipients:
                st.warning("No valid emails found to send alerts.")
                return False

            st.info("📧 Alert notifications will be sent to: " + ", ".join(recipients))


            st.info("🏥 **Immediate Resources:**")
            st.info("• National Suicide Prevention Lifeline: 988")
            st.info("• Crisis Text Line: Text HOME to 741741")
            st.info("• Emergency Services: 911")

            success = True
            for recipient in recipients:
                sent = self.send_email_alert(recipient, f"MindGuard AI ALERT for user {user_info['username']}", alert_message)
                success = success and sent

            return success
        except Exception as e:
            st.error(f"Alert system error: {e}")
            return False

    def _create_alert_message(self, username: str, reasons: list, results: dict) -> str:
        message = f"""
🚨 MINDGUARD AI ALERT 🚨

User: {username}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Alert Reasons:
- {', '.join(reasons)}

Assessment Results:
{json.dumps(results, indent=2)}

Immediate Resources:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911

This is an automated alert from MindGuard AI indicating potential mental health concerns.
Please reach out to the user immediately for support.
    """
        return message
    

