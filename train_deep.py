"""
Train the CNN-LSTM model on raw filtered EEG windows.
Uses a chronological split and class-weighted loss (preictal is the minority class).

Usage:
    python train_deep.py --data_dir processed/chb01 --epochs 30 --batch_size 32
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

from dataset import EEGDataset
from model import CNNLSTM


def chronological_split(X, y, test_frac=0.2):
    n = len(y)
    split = int(n * (1 - test_frac))
    return X[:split], X[split:], y[:split], y[split:]


def main(args):
    X = np.load(f"{args.data_dir}/X.npy")
    y = np.load(f"{args.data_dir}/y.npy")

    X_train, X_test, y_train, y_test = chronological_split(X, y)
    print(f"Train: {len(y_train)} (preictal={y_train.sum()}) | Test: {len(y_test)} (preictal={y_test.sum()})")

    train_ds = EEGDataset(X_train, y_train)
    test_ds = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels, n_samples = X.shape[1], X.shape[2]
    model = CNNLSTM(n_channels, n_samples).to(device)

    # class weights: preictal is rare, so weight it higher
    n_pos, n_neg = y_train.sum(), (y_train == 0).sum()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: train loss = {total_loss / len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    print("\n--- Test Results ---")
    print(classification_report(all_labels, preds))
    print("AUC:", roc_auc_score(all_labels, all_probs))

    torch.save(model.state_dict(), f"{args.data_dir}/cnn_lstm_model.pt")
    print(f"Model saved to {args.data_dir}/cnn_lstm_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
