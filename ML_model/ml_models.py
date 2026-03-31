import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, precision_recall_curve,
    average_precision_score
)
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for overnight runs
import joblib
import time


#change directories as needed absolute path was used here
base_path = 'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/'
save_models_at ='C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/models/'
save_plots_at = 'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/plots/'





def main():
    print("Loading ML dataset...")
    df = pd.read_parquet(base_path + 'data_merged/df_ml.parquet')

    # Separate old and new
    df_old = df[df['collected'] == 'old'].copy()
    df_new = df[df['collected'] == 'new'].copy()

    # Feature columns
    drop_cols = ['Category', 'collected']
    bool_cols = df.select_dtypes(
        include='bool'
    ).columns.tolist()
    drop_cols += bool_cols
    feature_cols = [
        c for c in df.columns if c not in drop_cols
    ]
    print(f"Number of features: {len(feature_cols)}")

    # Prepare data
    X_old = df_old[feature_cols].fillna(0)
    y_old = df_old['Category']
    X_new = df_new[feature_cols].fillna(0)
    y_new = df_new['Category']

    # Align columns
    X_new = X_new.reindex(
        columns=X_old.columns, fill_value=0
    )

    # Train/test split
    X_train, X_test_old, y_train, y_test_old = \
        train_test_split(
            X_old, y_old,
            test_size=0.2,
            random_state=42,
            stratify=y_old
        )

    print(f"\nTraining set: {len(X_train)}")
    print(f"Test set (old): {len(X_test_old)}")
    print(f"Test set (new): {len(X_new)}")

    # Scale for LR and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_old_scaled = scaler.transform(X_test_old)
    X_new_scaled = scaler.transform(X_new)

    # Convert scaled arrays back to DataFrames
    # needed for feature category comparison
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=feature_cols
    )
    X_test_old_scaled_df = pd.DataFrame(
        X_test_old_scaled, columns=feature_cols
    )

    #all the models will be saved for easier replication
    joblib.dump(scaler, save_models_at + 'scaler.pkl')

    results_old = []
    results_new = []
    train_times = []
    model_names_list = []

    #first model Random Forest
    print("\nTraining Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    train_times.append(rf_time)
    model_names_list.append('RF')
    print(f"Trained in {rf_time:.1f}s")

    r_old = evaluate_model(
        "Random Forest", rf, X_test_old, y_test_old
    )
    r_new = evaluate_model(
        "Random Forest", rf, X_new, y_new
    )
    results_old.append(r_old)
    results_new.append(r_new)

    plot_confusion_matrix(
        y_test_old, r_old['y_pred'],
        'RandomForest', 'Historical'
    )
    plot_confusion_matrix(
        y_new, r_new['y_pred'],
        'RandomForest', 'Modern'
    )
    
    joblib.dump(rf, save_models_at + 'rf_model.pkl')

    # Second Model Logistical regression
    print("\nTraining Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(
    max_iter=5000,
    random_state=42,
    n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    lr_time = time.time() - start
    train_times.append(lr_time)
    model_names_list.append('LR')
    print(f"Trained in {lr_time:.1f}s")

    r_old = evaluate_model(
        "Logistic Regression", lr,
        X_test_old_scaled, y_test_old
    )
    r_new = evaluate_model(
        "Logistic Regression", lr,
        X_new_scaled, y_new
    )
    results_old.append(r_old)
    results_new.append(r_new)

    plot_confusion_matrix(
        y_test_old, r_old['y_pred'],
        'LogisticRegression', 'Historical'
    )
    plot_confusion_matrix(
        y_new, r_new['y_pred'],
        'LogisticRegression', 'Modern'
    )
    
    joblib.dump(lr, save_models_at + 'lr_model.pkl')

    


    # KNN Model
    print("\nTraining KNN...")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_time = time.time() - start
    train_times.append(knn_time)
    model_names_list.append('KNN')
    print(f"Trained in {knn_time:.1f}s")

    r_old = evaluate_model(
        "KNN", knn, X_test_old_scaled, y_test_old
    )
    r_new = evaluate_model(
        "KNN", knn, X_new_scaled, y_new
    )
    results_old.append(r_old)
    results_new.append(r_new)

    plot_confusion_matrix(
        y_test_old, r_old['y_pred'],
        'KNN', 'Historical'
    )
    plot_confusion_matrix(
        y_new, r_new['y_pred'],
        'KNN', 'Modern'
    )
   
    joblib.dump(knn, save_models_at + 'knn_model.pkl')

    
    #creating plots

    plot_roc_curves(results_old, y_test_old, 'Historical')
    plot_roc_curves(results_new, y_new, 'Modern')
    plot_precision_recall(results_old, y_test_old, 'Historical')
    plot_precision_recall(results_new, y_new, 'Modern')
    plot_accuracy_comparison(results_old, results_new)
    plot_feature_importance(rf, feature_cols)
    plot_data_distribution(
        df_old[feature_cols],
        df_new[feature_cols]
    )

    # statistical tests
    print("\n Statistical Significance Tests ")
    run_mcnemar(
        y_test_old,
        results_old[0]['y_pred'],  # RF
        results_old[1]['y_pred'],  # LR
        'Random Forest', 'Logistic Regression'
    )
    run_mcnemar(
        y_test_old,
        results_old[0]['y_pred'],  # RF
        results_old[2]['y_pred'],  # KNN
        'Random Forest', 'KNN'
    )
    run_mcnemar(
        y_test_old,
        results_old[1]['y_pred'],  # LR
        results_old[2]['y_pred'],  # KNN
        'Logistic Regression', 'KNN'
    )

    #cross validation tests
    cv_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    run_cross_validation(cv_models, X_old, y_old)

    #degration analysis 
    degradation_analysis(results_old, results_new)

    # 
    ablation_study(
        X_train, y_train,
        X_test_old, y_test_old,
        feature_cols
    )

    #FP/FN and TP/TN graphs
    plot_feature_category_comparison(
        X_train, y_train,
        X_test_old, y_test_old,
        X_train_scaled_df, X_test_old_scaled_df,
        feature_cols
    )

   #Sumnary table
    def clean_results(results):
        return pd.DataFrame([{
            'model': r['model'],
            'accuracy': round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1': round(r['f1'], 4),
            'fpr': round(r['fpr'], 4),
            'fnr': round(r['fnr'], 4)
        } for r in results])

    summary_old = clean_results(results_old)
    summary_new = clean_results(results_new)

    print("\n FINAL RESULTS: Historical Test Data ")
    print(summary_old.to_string(index=False))
    print("\n FINAL RESULTS: Modern Data ")
    print(summary_new.to_string(index=False))

    #saving results to csv so it can be used to thr dissertation without the need to rerun the code
    summary_old.to_csv(
        save_models_at + 'results_historical.csv', index=False
    )
    summary_new.to_csv(
        save_models_at + 'results_modern.csv', index=False
    )

    print(f"\nAll done.")
    print(f"Plots saved to: {save_plots_at}")
    print(f"Models saved to: {save_models_at}")

#basic model evaluation
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    # False positive / negative rates
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['fpr'] = fp / (fp + tn)
    metrics['fnr'] = fn / (fn + tp)
    metrics['fp'] = fp
    metrics['fn'] = fn


    #checking and printing the results
    print(f"\n{'-'*50}")
    print(f"  {name}")
    
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1 Score:            {metrics['f1']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f} "
          f"({fp} legitimate sites wrongly blocked)")
    print(f"False Negative Rate: {metrics['fnr']:.4f} "
          f"({fn} phishing sites missed)")
    print(f"\nConfusion Matrix:")
    print(cm)

    return metrics


# TP, TN, FP and FN plots historical
def plot_confusion_matrix(y_test, y_pred, 
                           model_name, dataset_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legitimate', 'Phishing']
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{model_name} - {dataset_name}')
    plt.tight_layout()
    plt.savefig(
        save_plots_at + f'cm_{model_name}_{dataset_name}.png',
        dpi=150
    )
    plt.close()

#ROC plot
def plot_roc_curves(results, y_test, dataset_name):
    plt.figure(figsize=(8, 6))
    for r in results:
        if r['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                label=f"{r['model']} (AUC = {roc_auc:.3f})"
            )
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(
        save_plots_at + f'roc_{dataset_name}.png', dpi=150
    )
    plt.close()
    #saving the results
    print(f"Saved ROC curve: {dataset_name}")


def plot_precision_recall(results, y_test, dataset_name):
    plt.figure(figsize=(8, 6))
    for r in results:
        if r['y_prob'] is not None:
            precision, recall, _ = precision_recall_curve(
                y_test, r['y_prob']
            )
            ap = average_precision_score(y_test, r['y_prob'])
            plt.plot(
                recall, precision,
                label=f"{r['model']} (AP = {ap:.3f})"
            )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {dataset_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(
        save_plots_at + f'pr_curve_{dataset_name}.png', dpi=150
    )
    plt.close()
    print(f"Saved PR curve: {dataset_name}")

# TP, TN, FP and FN plots new and old togoteher
def plot_accuracy_comparison(results_old, results_new):
    models = [r['model'] for r in results_old]
    acc_old = [r['accuracy'] for r in results_old]
    acc_new = [r['accuracy'] for r in results_new]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width/2, acc_old, width,
        label='Historical Data', color='steelblue'
    )
    bars2 = ax.bar(
        x + width/2, acc_new, width,
        label='Modern Data', color='coral'
    )
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy: Historical vs Modern Data')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    plt.savefig(save_plots_at + 'accuracy_comparison.png', dpi=150)
    plt.close()
    print("Saved accuracy comparison chart")

#feature importance
def plot_feature_importance(rf_model, feature_cols):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance_df['feature'][::-1],
        importance_df['importance'][::-1],
        color='steelblue'
    )
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(save_plots_at + 'feature_importance.png', dpi=150)
    plt.close()

    importance_df.to_csv(
        save_models_at + 'feature_importance.csv', index=False
    )
    print("Saved feature importance")
    print("\nTop 15 features:")
    print(importance_df.to_string(index=False))

#degradation analysis 
def degradation_analysis(results_old, results_new):
    print("\n Performance Degradation Analysis ")
    degradation = []

    for r_old, r_new in zip(results_old, results_new):
        drop = r_old['accuracy'] - r_new['accuracy']
        pct_drop = (drop / r_old['accuracy']) * 100
        degradation.append({
            'model': r_old['model'],
            'historical_acc': round(r_old['accuracy'], 4),
            'modern_acc': round(r_new['accuracy'], 4),
            'absolute_drop': round(drop, 4),
            'pct_drop': round(pct_drop, 2)
        })
        print(f"{r_old['model']}:")
        print(f"  Historical: {r_old['accuracy']:.4f} → "
              f"Modern: {r_new['accuracy']:.4f}")
        print(f"  Degradation: {drop:.4f} ({pct_drop:.1f}%)")

    deg_df = pd.DataFrame(degradation)
    deg_df.to_csv(
        save_models_at + 'degradation_analysis.csv', index=False
    )
    return deg_df



# Statistical test
#McNemar
def run_mcnemar(y_test, pred1, pred2, 
                name1, name2):
    both_right = sum(
        (pred1 == y_test.values) & 
        (pred2 == y_test.values)
    )
    only1_right = sum(
        (pred1 == y_test.values) & 
        (pred2 != y_test.values)
    )
    only2_right = sum(
        (pred1 != y_test.values) & 
        (pred2 == y_test.values)
    )
    both_wrong = sum(
        (pred1 != y_test.values) & 
        (pred2 != y_test.values)
    )

    table = [[both_right, only1_right],
             [only2_right, both_wrong]]
    result = mcnemar(table, exact=True)

    print(f"\nMcNemar Test: {name1} vs {name2}")
    print(f"p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("Result: Difference IS statistically significant")
    else:
        print("Result: Difference is NOT statistically significant")

    return result.pvalue

#Cross valudation
def run_cross_validation(models_dict, X_old, y_old):
    print("\n 5-Fold Cross Validation ")
    cv_results = []

    for name, model in models_dict.items():
        scores = cross_val_score(
            model, X_old, y_old,
            cv=5, scoring='f1', n_jobs=-1
        )
        print(f"{name}: F1 = {scores.mean():.4f} "
              f"(+/- {scores.std():.4f})")
        cv_results.append({
            'model': name,
            'cv_f1_mean': round(scores.mean(), 4),
            'cv_f1_std': round(scores.std(), 4)
        })

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(
        save_models_at + 'cross_validation.csv', index=False
    )
    return cv_df


# ablation study answers RQ2
def ablation_study(X_train, y_train, 
                   X_test, y_test, feature_cols):
    print("\n Ablation Study ")

    url_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'url_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_at', 'num_question',
            'num_equals', 'num_ampersands', 'num_digits',
            'num_subdomains', 'has_ip', 'has_https', 'has_port'
        ]
    )]

    html_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'num_links', 'num_external_links',
            'num_internal_links', 'ratio_external',
            'num_forms', 'num_inputs', 'has_password',
            'num_scripts', 'num_external_scripts',
            'num_iframes', 'num_images', 'num_meta',
            'has_title', 'html_length'
        ]
    )]

    meta_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'has_updated', 'has_status', 'has_emails',
            'has_country', 'has_registrar',
            'has_name_servers', 'has_org', 'email_count'
        ]
    )]

    experiments = {
        'URL Only': url_cols,
        'HTML Only': html_cols,
        'Metadata Only': meta_cols,
        'URL + HTML': url_cols + html_cols,
        'URL + Metadata': url_cols + meta_cols,
        'HTML + Metadata': html_cols + meta_cols,
        'All Features': feature_cols,
    }

    results = []
    for exp_name, cols in experiments.items():
        if not cols:
            print(f"Skipping {exp_name} - no columns")
            continue
        rf_temp = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf_temp.fit(X_train[cols], y_train)
        y_pred = rf_temp.predict(X_test[cols])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({
            'experiment': exp_name,
            'num_features': len(cols),
            'accuracy': round(acc, 4),
            'f1': round(f1, 4)
        })
        print(f"{exp_name}: "
              f"acc={acc:.4f} f1={f1:.4f} "
              f"({len(cols)} features)")

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(
        save_models_at + 'ablation_study.csv', index=False
    )

    # Plot ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        ablation_df['experiment'],
        ablation_df['accuracy'],
        color='steelblue'
    )
    ax.set_ylabel('Accuracy')
    ax.set_title('Ablation Study - Random Forest')
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha='right')
    for i, v in enumerate(ablation_df['accuracy']):
        ax.text(i, v + 0.01, f'{v:.3f}',
                ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_plots_at + 'ablation_study.png', dpi=150)
    plt.close()
    print("Saved ablation study")

    return ablation_df

#FP/FN and TP/TN graphs
def plot_feature_category_comparison(
        X_train, y_train, X_test_old, y_test_old,
        X_train_scaled, X_test_old_scaled,
        feature_cols):

    url_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'url_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_at', 'num_question',
            'num_equals', 'num_ampersands', 'num_digits',
            'num_subdomains', 'has_ip', 'has_https', 'has_port'
        ]
    )]

    html_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'num_links', 'num_external_links',
            'num_internal_links', 'ratio_external',
            'num_forms', 'num_inputs', 'has_password',
            'num_scripts', 'num_external_scripts',
            'num_iframes', 'num_images', 'num_meta',
            'has_title', 'html_length'
        ]
    )]

    meta_cols = [c for c in feature_cols if any(
        c.startswith(x) for x in [
            'has_updated', 'has_status', 'has_emails',
            'has_country', 'has_registrar',
            'has_name_servers', 'has_org', 'email_count'
        ]
    )]

    categories = {
        'URL Only': url_cols,
        'HTML Only': html_cols,
        'Metadata Only': meta_cols,
        'Combined': feature_cols
    }

    model_configs = {
        'RF': (
            RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            X_train, X_test_old
        ),
        'LR': (
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_scaled, X_test_old_scaled
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=5),
            X_train_scaled, X_test_old_scaled
        )
    }

    results = {cat: {} for cat in categories}

    for cat_name, cols in categories.items():
        if not cols:
            continue
        print(f"\nFeature category: {cat_name}")
        for model_name, (model, X_tr, X_te) in \
                model_configs.items():

            # Get column indices for scaled data
            col_idx = [
                list(X_train.columns).index(c)
                for c in cols
                if c in X_train.columns
            ]

            if hasattr(X_tr, 'columns'):
                X_tr_cat = X_tr[cols]
                X_te_cat = X_te[cols]
            else:
                X_tr_cat = X_tr[:, col_idx]
                X_te_cat = X_te[:, col_idx]

            model.fit(X_tr_cat, y_train)
            y_pred = model.predict(X_te_cat)
            acc = accuracy_score(y_test_old, y_pred)
            results[cat_name][model_name] = acc
            print(f"  {model_name}: {acc:.4f}")

    # Plot
    cat_names = list(categories.keys())
    model_names = ['RF', 'LR', 'KNN']
    colors = ['steelblue', 'coral', 'green']
    x = np.arange(len(cat_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, color) in enumerate(
            zip(model_names, colors)):
        accs = [
            results[cat].get(model_name, 0)
            for cat in cat_names
        ]
        bars = ax.bar(
            x + i * width, accs, width,
            label=model_name, color=color
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Feature Category')
    ax.set_xticks(x + width)
    ax.set_xticklabels(cat_names)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        save_plots_at + 'feature_category_comparison.png',
        dpi=150
    )
    plt.close()
    print("Saved feature category comparison")

#data distribution exploration
def plot_data_distribution(df_old, df_new):
    key_features = [
        'url_length', 'num_dots',
        'html_length', 'num_scripts', 'has_https'
    ]
    
    # Only use features that exist in both
    key_features = [
        f for f in key_features
        if f in df_old.columns and f in df_new.columns
    ]

    fig, axes = plt.subplots(
        1, len(key_features),
        figsize=(15, 4)
    )
    if len(key_features) == 1:
        axes = [axes]

    for i, feat in enumerate(key_features):
        axes[i].hist(
            df_old[feat].clip(
                upper=df_old[feat].quantile(0.99)
            ),
            bins=30, alpha=0.5,
            label='Historical', color='steelblue'
        )
        axes[i].hist(
            df_new[feat].clip(
                upper=df_new[feat].quantile(0.99)
            ),
            bins=30, alpha=0.5,
            label='Modern', color='coral'
        )
        axes[i].set_title(feat)
        axes[i].legend(fontsize=7)

    plt.suptitle(
        'Feature Distribution: Historical vs Modern Data'
    )
    plt.tight_layout()
    plt.savefig(
        save_plots_at + 'data_distribution.png', dpi=150
    )
    plt.close()
    print("Saved data distribution plot")



if __name__ == "__main__":
    main()