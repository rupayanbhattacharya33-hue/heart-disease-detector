import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, classification_report,
    precision_score, recall_score
)
from sklearn.inspection import DecisionBoundaryDisplay
from xgboost import XGBClassifier
 
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)
 
# ── 1. Load & clean ───────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/heart.csv")
print(f"Shape before cleaning: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
 
df["ca"]   = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())
print(f"NaNs after fill: {df.isnull().sum().sum()}")
 
df["target"] = (df["target"] > 0).astype(int)
print(f"Target distribution:\n{df['target'].value_counts()}")
 
# Save full dataset for percentile comparison in app
joblib.dump(df, "models/full_dataset.pkl")
 
# ── 2. Correlation heatmap ────────────────────────────────────────
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(
    df.corr(), mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", linewidths=0.5, square=True,
    cbar_kws={"shrink": 0.8}, annot_kws={"size": 9}
)
plt.title("Feature Correlation Heatmap", fontsize=15, pad=15, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: charts/correlation_heatmap.png")
 
# ── 3. Split ──────────────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
 
# ── 4. Train models ───────────────────────────────────────────────
print("\nTraining models...")
 
lr  = LogisticRegression(max_iter=1000, random_state=42)
dt  = DecisionTreeClassifier(max_depth=5, random_state=42)
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, verbosity=0
)
 
lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
xgb.fit(X_train, y_train)
 
models = {
    "Logistic Regression": (lr,  X_test_scaled, X_train_scaled),
    "Decision Tree":       (dt,  X_test,        X_train),
    "XGBoost":             (xgb, X_test,        X_train),
}
 
results = {}
print(f"\n{'Model':<25} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Prec':>7} {'Recall':>7} {'CV-AUC':>9}")
print("-" * 72)
 
for name, (m, Xte, Xtr) in models.items():
    preds = m.predict(Xte)
    proba = m.predict_proba(Xte)[:, 1]
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    # 5-fold cross validation AUC
    cv_sc = cross_val_score(m, np.vstack([Xtr, Xte]),
                            np.concatenate([y_train, y_test]),
                            cv=5, scoring="roc_auc")
    results[name] = {
        "acc": acc, "f1": f1, "auc": auc,
        "prec": prec, "rec": rec, "cv_auc": cv_sc.mean(),
        "preds": preds, "proba": proba
    }
    print(f"{name:<25} {acc:>7.3f} {f1:>7.3f} {auc:>7.3f} "
          f"{prec:>7.3f} {rec:>7.3f} {cv_sc.mean():>9.3f}±{cv_sc.std():.3f}")
 
# ── 5. Save XGBoost ───────────────────────────────────────────────
joblib.dump(xgb, "models/xgb_model.pkl")
 
# Save cross-val scores for display in app
joblib.dump({k: v["cv_auc"] for k, v in results.items()},
            "models/cv_scores.pkl")
print("\nSaved: models/xgb_model.pkl")
 
# ── 6. Confusion matrix ───────────────────────────────────────────
print("Generating confusion matrix...")
xgb_preds = results["XGBoost"]["preds"]
cm = confusion_matrix(y_test, xgb_preds)
 
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"})
ax.set_title("XGBoost — Confusion Matrix", fontsize=14, pad=12, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
 
# Annotate quadrants
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, 2.35, f"True Negative: {tn} ✓", ha="center", fontsize=9, color="#16a34a")
ax.text(1.5, 2.35, f"False Positive: {fp} ✗", ha="center", fontsize=9, color="#dc2626")
ax.text(0.5, 2.55, f"False Negative: {fn} ✗", ha="center", fontsize=9, color="#dc2626")
ax.text(1.5, 2.55, f"True Positive: {tp} ✓", ha="center", fontsize=9, color="#16a34a")
 
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/confusion_matrix.png")
 
# ── 7. ROC curves ─────────────────────────────────────────────────
print("Generating ROC curves...")
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"Logistic Regression": "#378ADD",
          "Decision Tree":       "#1D9E75",
          "XGBoost":             "#D85A30"}
lws    = {"Logistic Regression": 2,
          "Decision Tree":       2,
          "XGBoost":             3}
 
for name, (m, Xte, _) in models.items():
    proba = results[name]["proba"]
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    auc = results[name]["auc"]
    ax.plot(fpr, tpr, color=colors[name], lw=lws[name],
            label=f"{name}  (AUC = {auc:.3f})")
    # Mark optimal threshold point
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx],
               color=colors[name], s=80, zorder=5)
 
ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random classifier (AUC = 0.500)")
ax.fill_between([0,1],[0,1],[1,1], alpha=0.03, color="green")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — All Models", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
plt.tight_layout()
plt.savefig("charts/roc_curve.png", dpi=150)
plt.close()
print("Saved: charts/roc_curve.png")
 
# ── 8. Model comparison ───────────────────────────────────────────
print("Generating model comparison chart...")
metrics     = ["acc", "f1", "auc", "prec", "rec"]
metric_lbls = ["Accuracy", "F1 Score", "ROC-AUC", "Precision", "Recall"]
model_names = list(results.keys())
bar_colors  = ["#378ADD", "#1D9E75", "#D85A30"]
 
fig, axes = plt.subplots(1, 5, figsize=(14, 4))
for i, (metric, title) in enumerate(zip(metrics, metric_lbls)):
    vals = [results[m][metric] for m in model_names]
    bars = axes[i].bar(model_names, vals, color=bar_colors, width=0.5,
                       edgecolor="white", linewidth=1.2)
    axes[i].set_title(title, fontsize=11, fontweight="bold")
    axes[i].set_ylim(0.65, 1.02)
    axes[i].set_xticklabels(
        [n.replace(" ", "\n") for n in model_names],
        fontsize=8
    )
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                     fontweight="bold")
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    axes[i].grid(axis="y", alpha=0.3)
 
plt.suptitle("Model Comparison — All Metrics", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/model_comparison.png")
 
# ── 9. SHAP feature importance ────────────────────────────────────
print("Calculating SHAP values...")
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
joblib.dump(explainer, "models/shap_explainer.pkl")
 
# Bar plot
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
 
# Dot/beeswarm plot
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_names, show=False)
plt.title("SHAP Summary — Feature Direction & Magnitude",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/shap_dot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/shap_importance.png + shap_dot.png")
 
# ── 10. Decision boundary (top 2 SHAP features) ──────────────────
print("Generating decision boundary chart...")
mean_shap = np.abs(shap_values).mean(axis=0)
top2_idx  = np.argsort(mean_shap)[-2:][::-1]
f1_name   = feature_names[top2_idx[0]]
f2_name   = feature_names[top2_idx[1]]
 
X2_train = X_train[[f1_name, f2_name]].values
X2_test  = X_test[[f1_name, f2_name]].values
 
from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf2.fit(X2_train, y_train)
 
fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(
    clf2, X2_train,
    response_method="predict_proba",
    plot_method="pcolormesh",
    xlabel=f1_name, ylabel=f2_name,
    alpha=0.25, ax=ax,
    cmap="RdYlGn_r"
)
scatter = ax.scatter(
    X2_test[:, 0], X2_test[:, 1],
    c=y_test, cmap="RdYlGn_r",
    edgecolors="white", linewidth=0.8,
    s=70, zorder=3
)
plt.colorbar(scatter, ax=ax, label="Disease (1) / No Disease (0)")
ax.set_title(f"Decision Boundary\n{f1_name}  vs  {f2_name}",
             fontsize=13, fontweight="bold", pad=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("charts/decision_boundary.png", dpi=150)
plt.close()
 
# Save top2 feature names for app
joblib.dump((f1_name, f2_name, clf2), "models/boundary_clf.pkl")
print("Saved: charts/decision_boundary.png")
 
# ── 11. Feature distribution by class ────────────────────────────
print("Generating feature distributions...")
top4_features = [feature_names[i] for i in np.argsort(mean_shap)[-4:][::-1]]
 
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
palette = {0: "#378ADD", 1: "#D85A30"}
labels  = {0: "No Disease", 1: "Disease"}
 
for i, feat in enumerate(top4_features):
    for cls in [0, 1]:
        subset = df[df["target"] == cls][feat]
        axes[i].hist(subset, bins=18, alpha=0.65,
                     color=palette[cls], label=labels[cls],
                     edgecolor="white", linewidth=0.5)
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Value", fontsize=9)
    axes[i].set_ylabel("Count", fontsize=9)
    axes[i].legend(fontsize=8)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    axes[i].grid(axis="y", alpha=0.3)
 
plt.suptitle("Feature Distributions by Class (Top 4 SHAP Features)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/feature_distributions.png")
 
# ── 12. Cross-validation scores chart ────────────────────────────
print("Generating cross-validation chart...")
cv_results = {}
for name, (m, Xte, Xtr) in models.items():
    scores = cross_val_score(
        m,
        np.vstack([Xtr, Xte]),
        np.concatenate([y_train, y_test]),
        cv=5, scoring="roc_auc"
    )
    cv_results[name] = scores
 
fig, ax = plt.subplots(figsize=(8, 5))
positions = np.arange(len(cv_results))
for i, (name, scores) in enumerate(cv_results.items()):
    ax.scatter([i]*5, scores, color=bar_colors[i], s=60, zorder=3, alpha=0.8)
    ax.plot([i-0.2, i+0.2], [scores.mean(), scores.mean()],
            color=bar_colors[i], lw=3, zorder=4)
    ax.errorbar(i, scores.mean(), yerr=scores.std(),
                color=bar_colors[i], capsize=6, lw=2, zorder=5)
    ax.text(i, scores.mean() - 0.025,
            f"μ={scores.mean():.3f}\nσ={scores.std():.3f}",
            ha="center", fontsize=9, color=bar_colors[i], fontweight="bold")
 
ax.set_xticks(positions)
ax.set_xticklabels(list(cv_results.keys()), fontsize=10)
ax.set_ylabel("ROC-AUC Score", fontsize=11)
ax.set_title("5-Fold Cross-Validation — ROC-AUC per Model",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.7, 1.02)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("charts/cross_validation.png", dpi=150)
plt.close()
print("Saved: charts/cross_validation.png")
 
# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("TRAINING COMPLETE")
print("="*55)
xr = results["XGBoost"]
print(f"Accuracy  : {xr['acc']:.3f}")
print(f"F1 Score  : {xr['f1']:.3f}")
print(f"ROC-AUC   : {xr['auc']:.3f}")
print(f"Precision : {xr['prec']:.3f}")
print(f"Recall    : {xr['rec']:.3f}")
print("\nDetailed classification report:")
print(classification_report(y_test, xgb_preds,
      target_names=["No Disease", "Disease"]))
print("\nAll charts saved in charts/")
print("All models saved in models/")