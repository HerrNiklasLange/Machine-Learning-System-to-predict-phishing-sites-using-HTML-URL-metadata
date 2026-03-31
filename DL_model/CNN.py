import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay, precision_recall_curve,
    average_precision_score
)
import matplotlib
import matplotlib.pyplot as plt
import time
import os


save_models_directory = 'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/models/'
save_plot_directory = 'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/plots/'
matplotlib.use('Agg') #need this as matplotlib wont we using a GUI


#Global for model limitations
# URL - character level
url_max_len = 200
url_vocab_size = 128  # ASCII characters

# HTML - word level
html_max_len = 500
html_vocab_size = 10000

# Metadata - word level
metadata_max_len = 100
metada_vocab_size = 5000

set_batch_size = 64
epochs = 10

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"Using device: {device}")


def main():
    print("Loading DL dataset...")
    df = pd.read_parquet(
        'C:/Users/nikla/OneDrive/Desktop/thesis-webscraper/data_merged/df_dl.parquet'
    )
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['Category'].value_counts()}")

    # Separate old and new
    df_old = df[df['collected'] == 'old'].copy()
    df_new = df[df['collected'] == 'new'].copy()

    print(f"\nOld data: {len(df_old)} rows")
    print(f"New data: {len(df_new)} rows")

    # Train/val/test split on old data
    from sklearn.model_selection import train_test_split
    df_train, df_test_old = train_test_split(
        df_old, test_size=0.2,
        random_state=42,
        stratify=df_old['Category']
    )
    df_train, df_val = train_test_split(
        df_train, test_size=0.1,
        random_state=42,
        stratify=df_train['Category']
    )

    print(f"\nTrain: {len(df_train)}")
    print(f"Validation: {len(df_val)}")
    print(f"Test (old): {len(df_test_old)}")
    print(f"Test (new): {len(df_new)}")

    
    print("\nFitting tokenizers on training data...")

    html_tokenizer = WordTokenizer(html_vocab_size)
    html_tokenizer.fit(df_train['html'].tolist())

    meta_tokenizer = WordTokenizer(metada_vocab_size)
    meta_tokenizer.fit(df_train['metadata_text'].tolist())

    #creating the dataset
    print("\nCreating datasets...")

    train_dataset = PhishingDataset(
        df_train['url'], df_train['html'],
        df_train['metadata_text'], df_train['Category'],
        html_tokenizer, meta_tokenizer
    )
    val_dataset = PhishingDataset(
        df_val['url'], df_val['html'],
        df_val['metadata_text'], df_val['Category'],
        html_tokenizer, meta_tokenizer
    )
    test_old_dataset = PhishingDataset(
        df_test_old['url'], df_test_old['html'],
        df_test_old['metadata_text'],
        df_test_old['Category'],
        html_tokenizer, meta_tokenizer
    )
    test_new_dataset = PhishingDataset(
        df_new['url'], df_new['html'],
        df_new['metadata_text'], df_new['Category'],
        html_tokenizer, meta_tokenizer
    )

    train_loader = DataLoader(
        train_dataset, set_batch_size=set_batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, set_batch_size=set_batch_size,
        shuffle=False, num_workers=0
    )
    test_old_loader = DataLoader(
        test_old_dataset, set_batch_size=set_batch_size,
        shuffle=False, num_workers=0
    )
    test_new_loader = DataLoader(
        test_new_dataset, set_batch_size=set_batch_size,
        shuffle=False, num_workers=0
    )

    #Start training CNN model
    print("\nInitialising CNN model...")
    model = PhishingCNN().to(device)
    print(model)

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Total parameters: {total_params:,}")

    optimizer = Adam(
        model.parameters(), lr=0.001 #setting learning rate
    )
    criterion = nn.BCELoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_labels, val_preds, _ = \
            evaluate_epoch(model, val_loader, criterion)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} "
              f"({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f} "
              f"Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} "
              f"Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                save_models_directory + 'cnn_best.pt'
            )
            print(f"  → Best model saved")

    #saving the best model
    print("\nLoading best model for evaluation...")
    model.load_state_dict(
        torch.load(save_models_directory + 'cnn_best.pt')
    )

    print("\nEvaluating on historical test data...")
    _, old_labels, old_preds, old_probs = \
        evaluate_epoch(model, test_old_loader, criterion)

    print("\nEvaluating on modern data...")
    _, new_labels, new_preds, new_probs = \
        evaluate_epoch(model, test_new_loader, criterion)

    results_old = full_evaluation(
        'Historical', old_labels, old_preds, old_probs
    )
    results_new = full_evaluation(
        'Modern', new_labels, new_preds, new_probs
    )

    # Plots creation


    plot_training_history(history)

    # ROC curve
    plt.figure(figsize=(8, 6))
    for result, y_test, name in [
        (results_old, old_labels, 'Historical'),
        (results_new, new_labels, 'Modern')
    ]:
        fpr, tpr, _ = roc_curve(
            y_test, result['y_prob']
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            label=f"CNN {name} (AUC = {roc_auc:.3f})"
        )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_plot_directory + 'roc_CNN.png', dpi=150)
    plt.close()

    # Degradation
    drop = results_old['accuracy'] - results_new['accuracy']
    pct = (drop / results_old['accuracy']) * 100
    print(f"\n CNN Degradation ")
    print(f"Historical: {results_old['accuracy']:.4f} → "
          f"Modern: {results_new['accuracy']:.4f}")
    print(f"Degradation: {drop:.4f} ({pct:.1f}%)")

    
    
    #Summary
    print("\n CNN FINAL RESULTS")
    print(f"Historical Test Data:")
    print(f"  Accuracy:  {results_old['accuracy']:.4f}")
    print(f"  Precision: {results_old['precision']:.4f}")
    print(f"  Recall:    {results_old['recall']:.4f}")
    print(f"  F1:        {results_old['f1']:.4f}")
    print(f"\nModern Data:")
    print(f"  Accuracy:  {results_new['accuracy']:.4f}")
    print(f"  Precision: {results_new['precision']:.4f}")
    print(f"  Recall:    {results_new['recall']:.4f}")
    print(f"  F1:        {results_new['f1']:.4f}")

    # Save results
    pd.DataFrame([{
        'dataset': 'Historical',
        'accuracy': round(results_old['accuracy'], 4),
        'precision': round(results_old['precision'], 4),
        'recall': round(results_old['recall'], 4),
        'f1': round(results_old['f1'], 4),
        'fpr': round(results_old['fpr'], 4),
        'fnr': round(results_old['fnr'], 4)
    }, {
        'dataset': 'Modern',
        'accuracy': round(results_new['accuracy'], 4),
        'precision': round(results_new['precision'], 4),
        'recall': round(results_new['recall'], 4),
        'f1': round(results_new['f1'], 4),
        'fpr': round(results_new['fpr'], 4),
        'fnr': round(results_new['fnr'], 4)
    }]).to_csv(save_models_directory + 'cnn_results.csv', index=False)

    print("\nCNN training complete.")
    print(f"Best model: {save_models_directory}cnn_best.pt")
    print(f"Plots: {save_plot_directory}")


#Openining and then tokenizing dataset
class PhishingDataset(Dataset):
    def __init__(self, urls, htmls, metas, labels,
                 html_tokenizer, meta_tokenizer):
        self.labels = torch.tensor(
            labels.values, dtype=torch.float32
        )

        print("Tokenizing URLs (character level)...")
        self.url_tokens = torch.tensor([
            char_tokenize(url, url_max_len)
            for url in urls
        ], dtype=torch.long)

        print("Tokenizing HTML (word level)...")
        self.html_tokens = torch.tensor([
            html_tokenizer.tokenize(html, html_max_len)
            for html in htmls
        ], dtype=torch.long)

        print("Tokenizing metadata (word level)...")
        self.meta_tokens = torch.tensor([
            meta_tokenizer.tokenize(meta, metadata_max_len)
            for meta in metas
        ], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.url_tokens[idx],
            self.html_tokens[idx],
            self.meta_tokens[idx],
            self.labels[idx]
        )

#WordTokenizer for HTML and metadata
class WordTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.fitted = False

    def fit(self, texts):
        from collections import Counter
        all_words = []
        for text in texts:
            all_words.extend(str(text).lower().split())
        word_counts = Counter(all_words)
        
        top_words = word_counts.most_common(
            self.vocab_size - 2
        )
        for i, (word, _) in enumerate(top_words, start=2):
            self.word2idx[word] = i
        self.fitted = True
        print(f"Vocabulary size: {len(self.word2idx)}")

    def tokenize(self, text, max_len):
        words = str(text).lower().split()[:max_len]
        tokens = [
            self.word2idx.get(w, 1) for w in words
        ]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return tokens[:max_len]
#CharacterTokenizer for URL
def char_tokenize(text, max_len, vocab_size=128):
    text = str(text)[:max_len]
    tokens = [min(ord(c), vocab_size - 1) for c in text]
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]


class PhishingCNN(nn.Module):
    def __init__(self):
        super(PhishingCNN, self).__init__()

        # --- URL branch (character level) ---
        self.url_embedding = nn.Embedding(
            url_vocab_size, 64, padding_idx=0
        )
        self.url_conv1 = nn.Conv1d(64, 128, kernel_size=3,
                                    padding=1)
        self.url_conv2 = nn.Conv1d(128, 64, kernel_size=3,
                                    padding=1)
        self.url_pool = nn.AdaptiveMaxPool1d(1)

        # --- HTML branch (word level) ---
        self.html_embedding = nn.Embedding(
            html_vocab_size, 128, padding_idx=0
        )
        self.html_conv1 = nn.Conv1d(128, 256, kernel_size=3,
                                     padding=1)
        self.html_conv2 = nn.Conv1d(256, 128, kernel_size=3,
                                     padding=1)
        self.html_pool = nn.AdaptiveMaxPool1d(1)

        # --- Metadata branch (word level) ---
        self.meta_embedding = nn.Embedding(
            metada_vocab_size, 64, padding_idx=0
        )
        self.meta_conv1 = nn.Conv1d(64, 128, kernel_size=3,
                                     padding=1)
        self.meta_conv2 = nn.Conv1d(128, 64, kernel_size=3,
                                     padding=1)
        self.meta_pool = nn.AdaptiveMaxPool1d(1)

        # --- Combined classifier ---
        # 64 + 128 + 64 = 256 from three branches
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, url, html, meta):
        # URL branch
        u = self.url_embedding(url)
        u = u.permute(0, 2, 1)
        u = self.relu(self.url_conv1(u))
        u = self.relu(self.url_conv2(u))
        u = self.url_pool(u).squeeze(-1)

        # HTML branch
        h = self.html_embedding(html)
        h = h.permute(0, 2, 1)
        h = self.relu(self.html_conv1(h))
        h = self.relu(self.html_conv2(h))
        h = self.html_pool(h).squeeze(-1)

        # Metadata branch
        m = self.meta_embedding(meta)
        m = m.permute(0, 2, 1)
        m = self.relu(self.meta_conv1(m))
        m = self.relu(self.meta_conv2(m))
        m = self.meta_pool(m).squeeze(-1)

        # Concatenate all branches
        combined = torch.cat([u, h, m], dim=1)
        combined = self.dropout(combined)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1)


#Training the model
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for url, html, meta, labels in loader:
        url = url.to(device)
        html = html.to(device)
        meta = meta.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(url, html, meta)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


#Evaluation of each epoch
def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for url, html, meta, labels in loader:
            url = url.to(device)
            html = html.to(device)
            meta = meta.to(device)
            labels = labels.to(device)

            outputs = model(url, html, meta)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(float)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        all_labels, all_preds, all_probs
    )


#plotting
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_plot_directory + 'cnn_training_history.png',
                dpi=150)
    plt.close()
    print("Saved CNN training history")


#Final evaluation
def full_evaluation(name, labels, preds, probs):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'--'*50}")
    print(f"  CNN - {name}")
    
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds):.4f}")
    print(f"Recall:    {recall_score(labels, preds):.4f}")
    print(f"F1 Score:  {f1_score(labels, preds):.4f}")
    print(f"FPR: {fp/(fp+tn):.4f} "
          f"({fp} legitimate wrongly blocked)")
    print(f"FNR: {fn/(fn+tp):.4f} "
          f"({fn} phishing missed)")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legitimate', 'Phishing']
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'CNN - {name}')
    plt.tight_layout()
    plt.savefig(
        save_plot_directory + f'cm_CNN_{name}.png', dpi=150
    )
    plt.close()

    return {
        'model': 'CNN',
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'fpr': fp / (fp + tn),
        'fnr': fn / (fn + tp),
        'y_pred': preds,
        'y_prob': probs
    }


if __name__ == "__main__":
    main()