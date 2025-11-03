# =============================
# CRISP-DM: Business Understanding
# =============================
# Goal: Predict car price, identify key drivers, and extract actionable insights for a dealership.

# =============================
# Imports & Setup
# =============================
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", rc={"figure.figsize": (10,6)})

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Helper metrics
def eval_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# =============================
# Data Understanding
# =============================
df = pd.read_csv("../data/used_cars.csv")   # adjust path if needed
print(df.head())
print(df.shape)
print(df.isna().mean().sort_values(ascending=False).head(10))

# Keep a focused subset of commonly available columns
keep_cols = [
    "price","year","manufacturer","condition","cylinders","fuel","odometer",
    "title_status","transmission","drive","size","type","paint_color","state"
]
df = df[[c for c in keep_cols if c in df.columns]].copy()

# Basic sanity filters
df = df[df["price"].between(500, 150000)]   # drop ridiculous prices
if "year" in df.columns:
    df = df[df["year"].between(1985, 2025)]
if "odometer" in df.columns:
    df = df[df["odometer"].between(0, 400000)]

# Drop rows missing target or core predictors
df = df.dropna(subset=["price"])
print("Rows after cleaning:", df.shape[0])

# =============================
# Quick EDA (Descriptive)
# =============================
fig, ax = plt.subplots()
sns.histplot(df["price"], bins=60, kde=True, ax=ax)
ax.set_title("Price Distribution")
ax.set_xlabel("Price")
plt.show()

if "odometer" in df.columns:
    ax = sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42),
                         x="odometer", y="price", alpha=0.4)
    ax.set_title("Price vs Odometer")
    plt.show()

if "year" in df.columns:
    ax = sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42),
                         x="year", y="price", alpha=0.4)
    ax.set_title("Price vs Year")
    plt.show()

for cat in ["manufacturer","condition","transmission","fuel","type"]:
    if cat in df.columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=df.sample(min(6000, len(df)), random_state=42), x=cat, y="price")
        plt.title(f"Price by {cat}")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

# =============================
# Data Preparation
# =============================
target = "price"
y = df[target].values

# Identify columns
num_cols = [c for c in ["year","odometer","cylinders"] if c in df.columns]
cat_cols = [c for c in df.columns if c not in num_cols + [target]]

# Handle missing values (simple): fill NA in numeric with median; categorical with 'missing'
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna("missing")

X = df[num_cols + cat_cols]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
numeric_tf   = Pipeline(steps=[("scaler", StandardScaler())])
categorical_tf = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ]
)

# =============================
# Modeling — Baselines & CV
# =============================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_report(model, name):
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    print(f"{name} | CV R2: {scores.mean():.3f} ± {scores.std():.3f}")

# Baselines
cv_report(LinearRegression(), "Linear Regression")
cv_report(RidgeCV(alphas=np.logspace(-3,3,25)), "RidgeCV")
cv_report(LassoCV(alphas=np.logspace(-3,1,25), max_iter=5000, random_state=42), "LassoCV")

# Tree Ensembles (with quick defaults)
cv_report(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), "RandomForest (default)")
cv_report(GradientBoostingRegressor(random_state=42), "GradientBoosting (default)")

# =============================
# Grid Search (RF & GB)
# =============================
rf_pipe = Pipeline(steps=[("prep", preprocess),
                         ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])

rf_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12, 18],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_gs.fit(X_train, y_train)
print("RF best params:", rf_gs.best_params_)
print("RF best CV RMSE:", -rf_gs.best_score_)

gb_pipe = Pipeline(steps=[("prep", preprocess),
                         ("model", GradientBoostingRegressor(random_state=42))])

gb_grid = {
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0]
}

gb_gs = GridSearchCV(gb_pipe, gb_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
gb_gs.fit(X_train, y_train)
print("GB best params:", gb_gs.best_params_)
print("GB best CV RMSE:", -gb_gs.best_score_)

# =============================
# Evaluation on Test Set
# =============================
candidates = {
    "Linear": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
    "RidgeCV": Pipeline([("prep", preprocess), ("model", RidgeCV(alphas=np.logspace(-3,3,25)))]),
    "LassoCV": Pipeline([("prep", preprocess), ("model", LassoCV(alphas=np.logspace(-3,1,25), max_iter=5000, random_state=42))]),
    "RF_best": rf_gs.best_estimator_,
    "GB_best": gb_gs.best_estimator_
}

test_scores = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_scores[name] = eval_regression(y_test, pred)

pd.DataFrame(test_scores).T.sort_values("RMSE")
