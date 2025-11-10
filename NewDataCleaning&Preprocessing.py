import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# 1. Load dataset
# -------------------------
DATA_PATH = "digital_wellbeing_dataset.csv"  # adjust if needed
assert os.path.exists(DATA_PATH), f"File not found at {DATA_PATH}. Upload dataset to this path."

df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False).head(20))

# -------------------------
# 2. Target detection / construction
# -------------------------
def detect_target(df):
    # common mental health columns to look for:
    candidates = ['mental_health', 'mental_health_score', 'wellbeing_score', 'stress', 'stress_level',
                  'anxiety', 'depression', 'anxiety_level', 'depression_level', 'mh_risk', 'risk_level']
    found = [c for c in df.columns if c.lower() in candidates or any(k in c.lower() for k in candidates)]
    return found

found_targets = detect_target(df)
print("\nDetected direct candidate target columns:", found_targets)

if found_targets:
    target_col = found_targets[0]
    print(f"Using detected target column: {target_col}")
    y_raw = df[target_col]
else:
    # If no direct target, try to construct a composite risk from anxiety/stress/depression numeric columns
    print("No direct target found — attempting to construct a composite risk target from numeric mental-health indicators.")
    mental_cols = [c for c in df.columns if any(k in c.lower() for k in ['anxiety', 'stress', 'depress', 'wellbeing', 'mh', 'mental'])]
    mental_num = [c for c in mental_cols if pd.api.types.is_numeric_dtype(df[c])]
    print("Numeric mental-health-related cols found:", mental_num)
    if len(mental_num) >= 1:
        # create composite score = mean of z-scores of these columns (after imputing)
        df_mental = df[mental_num].copy()
        df_mental = df_mental.fillna(df_mental.mean())  # simple impute for composite creation
        z = (df_mental - df_mental.mean())/df_mental.std(ddof=0)
        df['mh_composite'] = z.mean(axis=1)
        # bin into 3 categories (Low/Medium/High) using quantiles
        df['mh_risk'] = pd.qcut(df['mh_composite'], q=3, labels=['Low','Medium','High'])
        target_col = 'mh_risk'
        y_raw = df[target_col]
        print("Constructed target 'mh_risk' (Low/Medium/High). Distribution:")
        print(df[target_col].value_counts())
    else:
        # fallback: if absolutely no mental columns, try to use a generic 'label' column if present
        if 'label' in df.columns:
            target_col = 'label'
            y_raw = df[target_col]
            print("Using 'label' column as target.")
        else:
            raise ValueError("No suitable target found or constructible. Please provide a target column like 'stress', 'anxiety', 'wellbeing_score', or let me know how to derive the label.")

# If y_raw is numeric (regression-like), convert to categorical bins if classification goal desired
if pd.api.types.is_numeric_dtype(y_raw):
    print(f"Target '{target_col}' is numeric — binning into 3 categories (Low/Medium/High).")
    df[target_col + "_binned"] = pd.qcut(y_raw, 3, labels=['Low','Medium','High'])
    target_col = target_col + "_binned"

# Final target
print("Final target column:", target_col)
print(df[target_col].value_counts())

# -------------------------
# 3. Feature selection
# -------------------------
# Exclude target and obvious identifiers
excluded = [target_col]
id_like = [c for c in df.columns if any(k in c.lower() for k in ['id','user','timestamp','date','session'])]
print("Identifier-like cols to exclude:", id_like)
excluded += id_like

X = df.drop(columns=excluded)
y = df[target_col].astype(str)  # ensure string labels for classification

# Separate numeric and categorical predictors
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# If too many categorical unique values, consider dropping high-cardinality columns
high_card = [c for c in categorical_features if X[c].nunique() > 50]
if high_card:
    print("High-cardinality columns (will be dropped):", high_card)
    X = X.drop(columns=high_card)
    categorical_features = [c for c in categorical_features if c not in high_card]

# -------------------------
# 4. Preprocessing pipelines (Fixed)
# -------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # <-- fixed here
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop'
)
# -------------------------
# 5. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train label distribution:\n", y_train.value_counts(normalize=True))

# -------------------------
# 6. Models + pipelines + hyperparam grids
# -------------------------
models = {
    'LogisticRegression': (LogisticRegression(max_iter=2000, solver='liblinear', multi_class='ovr'),
                           {'clf__C':[0.01,0.1,1,10]}),
    'KNeighbors': (KNeighborsClassifier(),
                   {'clf__n_neighbors':[3,5,7], 'clf__weights':['uniform','distance']}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42),
                     {'clf__max_depth':[3,5,8,None], 'clf__min_samples_split':[2,5,10]}),
    'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=-1),
                     {'clf__n_estimators':[100,200], 'clf__max_depth':[5,10,None], 'clf__min_samples_split':[2,5]})
}

results = {}

# We'll use SMOTE to handle class imbalance inside an imblearn Pipeline
for name, (clf, grid) in models.items():
    print("\n==== Training", name, "====")
    # Build pipeline: preprocessing -> SMOTE -> classifier
    pipe = ImbPipeline(steps=[
        ('preproc', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])
    # small grid search with StratifiedKFold
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)
    print("Best cv accuracy:", gs.best_score_)
    # evaluate on test
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {name}: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    results[name] = {
        'model': best,
        'best_params': gs.best_params_,
        'cv_score': gs.best_score_,
        'test_acc': acc,
        'y_pred': y_pred
    }
    # Save model
    joblib.dump(best, f"{name}_best_pipeline.pkl")
    print(f"Saved pipeline to {name}_best_pipeline.pkl")

# -------------------------
# 7. Comparison plot
# -------------------------
res_df = pd.DataFrame([
    {'model': name, 'test_acc': info['test_acc'], 'cv_score': info['cv_score']}
    for name, info in results.items()
]).sort_values('test_acc', ascending=False)

print("\nModel comparison:")
print(res_df)

plt.figure(figsize=(8,4))
sns.barplot(data=res_df, x='model', y='test_acc')
plt.ylim(0,1)
plt.title("Test Accuracy by Model")
plt.ylabel("Accuracy")
plt.xlabel("")
plt.show()

# -------------------------
# 8. Confusion matrix for best model
# -------------------------
best_name = res_df.iloc[0]['model']
best_info = results[best_name]
best_model = best_info['model']
y_pred_best = best_info['y_pred']

cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {best_name}")
plt.show()

# -------------------------
# 9. If accuracy < target, suggestions printed
# -------------------------
target_accuracy = 0.90
best_acc = res_df['test_acc'].max()
if best_acc < target_accuracy:
    print(f"\nBest test accuracy = {best_acc:.4f} which is below target {target_accuracy:.2f}.")
    print("Suggestions to improve performance:")
    print("- Add richer features (temporal patterns, rolling averages, session segmentation).")
    print("- Increase dataset size or label quality.")
    print("- Try ensemble stacking or XGBoost/LightGBM (more powerful boosted trees).")
    print("- Use feature selection or dimensionality reduction (PCA) if many features.")
    print("- Check for label noise and balance class distributions.")
else:
    print(f"\nTarget reached! Best test accuracy = {best_acc:.4f}")

# -------------------------
# 10. Export best model and a small predict function
# -------------------------
def predict_single(sample_df, model_path=None):
    """Given a dataframe with same columns as original X, return prediction using saved pipeline"""
    if model_path is None:
        model = best_model
    else:
        model = joblib.load(model_path)
    return model.predict(sample_df)

# Example usage:
# sample = X_test.iloc[:3]
# print(predict_single(sample))

print("\nPipelines and models saved to current working directory.")
print("Next: use these saved pipelines in a web app (Flask or Dash) to serve predictions and plots.")

# Save summary CSV
res_df.to_csv("model_comparison_summary.csv", index=False)
print("Saved model_comparison_summary.csv")

# ============================================================
# 11. TRAIN LIGHTWEIGHT MODEL FOR USER DASHBOARD
# ============================================================
print("\n🔁 Training simplified model for user dashboard...")

# Select only features that match your Streamlit input fields
selected_features = [
    'Age',
    'Sleep_Hours',
    'Screen_Time_Hours',
    'Gaming_Hours',
    'Social_Media_Usage_Hours',
    'Stress_Level',
    'Physical_Activity_Hours',
    'Support_Systems_Access',
    'Online_Support_Usage',
    'Work_Environment_Impact'
]

# Keep only columns that actually exist in dataset
selected_features = [f for f in selected_features if f in df.columns]
print("Selected features available for retraining:", selected_features)

X_simplified = df[selected_features].copy()
y_simplified = y.copy()

# Split again
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simplified, y_simplified, test_size=0.2, random_state=42, stratify=y_simplified
)

# Preprocessing for simple model
num_features_s = X_simplified.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features_s = X_simplified.select_dtypes(include=['object', 'category']).columns.tolist()

num_tf_s = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_tf_s = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preproc_s = ColumnTransformer([
    ('num', num_tf_s, num_features_s),
    ('cat', cat_tf_s, cat_features_s)
])

# Simple Random Forest model for dashboard
rf_simple = RandomForestClassifier(n_estimators=200, random_state=42)

pipe_simple = ImbPipeline([
    ('preproc', preproc_s),
    ('smote', SMOTE(random_state=42)),
    ('clf', rf_simple)
])

pipe_simple.fit(X_train_s, y_train_s)
y_pred_s = pipe_simple.predict(X_test_s)
acc_s = accuracy_score(y_test_s, y_pred_s)

print(f"Simplified dashboard model accuracy: {acc_s:.3f}")
print("Classification report:\n", classification_report(y_test_s, y_pred_s))

# Save the lightweight model
joblib.dump(pipe_simple, "RandomForest_user_pipeline.joblib")
print("✅ Saved simplified model to RandomForest_user_pipeline.joblib")
