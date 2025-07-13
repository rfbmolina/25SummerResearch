from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# resamples the training fold to fix class imbalance
# standard-scales the features
# optionally compresses them with PCA (default: 50 comps ≈ 85-90 % var)
# fits scikit-learn compatible classifier
# evaluates on the untouched test fold (no leakage)
# The function is intentionally generic (sampler, scaler, PCA, classifier) are injected via arguments so you can grid-search different combinations from main().

# classifier: final estimator (already initialised), sampler: any imblearn sampler (or None: pass-through), name: used for prints / filenames, ncomponnets: PCA dimensionality 
def run_pipeline(X_tr, X_te, y_tr, y_te, classifier, sampler, name: str, random_state: int, ncomponents: int=50): 
    print(f"\n{name}")
    print("  before sampling  :", Counter(y_tr))

    # push the same seed into every step that supports it (makes the whole chain exactly reproducible)
    if hasattr(classifier, 'random_state'):
        classifier.random_state = random_state
    if sampler:
        if hasattr(sampler, 'random_state'):
            sampler.random_state = random_state

    
    #build the imblearn Pipeline. order matters: sampler-scaler-pca-classifier
    pipe = Pipeline([
    ("sampler", sampler),                         # fit_resample: returns X_res, y_res
    ("scaler",  StandardScaler()),                # fit/transform: returns X_scaled
    ("pca",     PCA(n_components=ncomponents,     # fit/transform: returns X_pca
                     random_state=random_state)),
    ("clf",     classifier),                      # fit: final estimator
])
    #Fit on training data only
    pipe.fit(X_tr, y_tr)

    #Diagnostic: feature count before/after PCA
    original_feature_count = X_tr.shape[1]
    reduced_feature_count = pipe.named_steps["pca"].n_components_

    print(f"  features before PCA : {original_feature_count}")
    print(f"  features after PCA  : {reduced_feature_count}")

    #Diagnostic: class balance after resampling (if sampler keeps indices)
    sampler_step = pipe.named_steps["sampler"]
    if hasattr(sampler_step, "sample_indices_"):
        idx_resampled = sampler_step.sample_indices_
        print("  after sampling   :", Counter(y_tr.iloc[idx_resampled]))
    else:
        print("  after sampling   : [not available for this sampler]")

    #predict on the untouched test fold
    y_pred = pipe.predict(X_te) # hard labels
    y_prob = pipe.predict_proba(X_te)[:, 1] # class 1 probabilities

    #confusion matrix + plots (saved to disk for later inspection)
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close()

    # Print additional metrics
    precision = precision_score(y_te, y_pred)
    recall = recall_score(y_te, y_pred)
    accuracy = (y_te == y_pred).mean()

    print(f"  Accuracy          : {accuracy:.3f}")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1-score          : {f1_score(y_te, y_pred):.3f}")
    print(f"  ROC-AUC           : {roc_auc_score(y_te, y_prob):.3f}")

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_te, y_prob)
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.grid(True)
    plt.savefig(f'precision_recall_curve_{name}.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve - {name} (AUC = {roc_auc:.3f})')
    plt.grid(True)
    plt.savefig(f'roc_curve_{name}.png')
    plt.close()
