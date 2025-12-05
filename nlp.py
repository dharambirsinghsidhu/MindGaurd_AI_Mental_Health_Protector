import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset from current directory
df = pd.read_csv('Combined_Data.csv')

# Remove rows with missing or empty labels
df = df[df['status'].notnull() & (df['status'].str.strip() != '')]

# Fill NaN in statement text to avoid errors
df['statement'] = df['statement'].fillna('')

# Encode target labels as numbers
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'], df['status_encoded'],
    test_size=0.2, random_state=42, stratify=df['status_encoded']
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# LightGBM classifier wrapper to work with scikit-learn ensemble
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb

from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb

class LGBMWrapper(BaseEstimator, ClassifierMixin):
    # Add this line to explicitly identify the class as a classifier
    _estimator_type = "classifier"

    def __init__(self, **params):
        self.params = params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        # It's good practice to store the classes seen during fit
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
# Initialize individual classifiers
lgb_clf = lgb.LGBMClassifier(num_leaves=31, n_estimators=200, learning_rate=0.1, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
lr_clf = LogisticRegression(max_iter=1000, random_state=42)

# Setup soft voting ensemble classifier
ensemble = VotingClassifier(
    estimators=[('lgb', lgb_clf), ('rf', rf_clf), ('lr', lr_clf)],
    voting='soft',
    n_jobs=-1
)

# Train ensemble model
ensemble.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = ensemble.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save all important components to current directory
joblib.dump(ensemble, 'ensemble_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Ensemble model, vectorizer, and label encoder saved successfully.")