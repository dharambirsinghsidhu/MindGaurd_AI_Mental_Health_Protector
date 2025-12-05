import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset from current directory
df = pd.read_csv('anxiety_depression_data.csv')

# Define target columns
target_columns = ['Anxiety_Score', 'Depression_Score', 'Stress_Level']

# Prepare features by dropping targets
X = df.drop(columns=target_columns)

# Encode categorical columns to numbers
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Separate targets
y_anxiety = df['Anxiety_Score']
y_depression = df['Depression_Score']
y_stress = df['Stress_Level']

# Split the data into training and testing sets
X_train, X_test, y_anxiety_train, y_anxiety_test = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)
_, _, y_depression_train, y_depression_test = train_test_split(X, y_depression, test_size=0.2, random_state=42)
_, _, y_stress_train, y_stress_test = train_test_split(X, y_stress, test_size=0.2, random_state=42)

# Define hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

def tune_and_train(X_train, y_train, X_test, y_test, target_name):
    print(f"\nTuning Random Forest for {target_name}...")

    rf = RandomForestRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    random_search.fit(X_train, y_train)

    print(f"Best parameters for {target_name}:", random_search.best_params_)

    best_rf = random_search.best_estimator_

    y_pred = best_rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    print(f"{target_name} MAE: {mae:.3f}")
    

    # Save the best model in the current directory
    filename = f'best_rf_{target_name.lower()}.pkl'
    joblib.dump(best_rf, filename)
    print(f"Saved best model for {target_name} as {filename}")

    return best_rf

# Tune and train models for each target variable
best_rf_anxiety = tune_and_train(X_train, y_anxiety_train, X_test, y_anxiety_test, "Anxiety_Score")
best_rf_depression = tune_and_train(X_train, y_depression_train, X_test, y_depression_test, "Depression_Score")
best_rf_stress = tune_and_train(X_train, y_stress_train, X_test, y_stress_test, "Stress_Level")

print("\nAll models have been tuned and saved successfully.")
