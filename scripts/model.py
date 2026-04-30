from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

def run_models(df):
    feature_cols = ['precipitation_mm', 'ivt', 'drought_severity_score',
                    'dry_land_memory', 'is_ar', 'extreme_precip']

    X = df[feature_cols].values
    y = df['is_whiplash'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=50,
        random_state=42,
        class_weight='balanced'
    )
    dt_model.fit(X_train_scaled, y_train)

    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    gb_auc = roc_auc_score(y_test, gb_pred_proba)

    os.makedirs("models", exist_ok=True)

    best_model = rf_model if rf_auc >= gb_auc else gb_model

    with open("models/best_classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return rf_auc, gb_auc
