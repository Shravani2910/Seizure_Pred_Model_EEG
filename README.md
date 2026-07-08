# EEG Seizure Prediction

Predict an upcoming seizure from scalp EEG **before** it happens, by classifying
short EEG windows as **preictal** (leading up to a seizure) vs **interictal**
(normal, seizure-free baseline).

---

## 1. Dataset: CHB-MIT Scalp EEG Database

- **Source**: PhysioNet — https://physionet.org/content/chbmit/1.0.0/
- **What it is**: 24 pediatric patients, continuous scalp EEG (23 channels,
  256 Hz), with seizure onset/offset times annotated in `.seizures` files.
- **Why this one**: it's the standard benchmark for seizure prediction papers,
  so you can directly compare your results to published work (typical
  sensitivity ~90%+, AUC ~0.85-0.95 depending on patient and method). It's also
  free, well-documented, and reasonably sized (~40GB total, but you can start
  with 2-3 patients, ~1-2GB each).

### How to download
```bash
# Whole dataset (large) — or grab one patient folder at a time, e.g. chb01
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/chb01/
```
Start with **chb01, chb03, chb05** — patients commonly used in papers, with a
decent number of seizures each.

### Labels you need to construct yourself
The dataset gives you seizure **start/end times**, not preictal/interictal
labels directly. You define:
- **Preictal window**: e.g. last 30–60 minutes before seizure onset (excluding
  a short "horizon" buffer right before onset, since that's basically ictal).
- **Interictal window**: EEG from segments at least several hours away from
  any seizure (so the model isn't accidentally learning postictal patterns).

This labeling step is in `src/preprocess.py`.

---

## 2. Plan / Steps

1. **Download** 2-3 patients' worth of `.edf` files + seizure annotation files.
2. **Preprocess**: load EDF with `mne`, band-pass filter (0.5-40 Hz), notch
   filter (50/60 Hz), segment into fixed-length windows (e.g. 10s, 50% overlap).
3. **Label** each window preictal / interictal based on distance to seizure onset.
4. **Feature extraction** (for the classical-ML baseline): band power
   (delta/theta/alpha/beta/gamma per channel), Hjorth parameters, statistical
   features. This is `src/features.py`.
5. **Baseline model**: Random Forest / XGBoost on extracted features —
   fast to train, surprisingly strong, good for a first working result.
6. **Deep learning model**: 1D-CNN + LSTM directly on raw filtered signal
   (no hand-crafted features) — this is the "real" model for your portfolio.
   `src/model.py` + `src/train.py`.
7. **Evaluation**: patient-specific, **leave-one-seizure-out** or chronological
   split (never randomly shuffle EEG windows — adjacent windows leak
   information into each other). Report sensitivity, false-prediction-rate
   per hour, and AUC — these are the standard seizure-prediction metrics.
8. **(Stretch)** Wrap it in a small Streamlit/Gradio demo that takes an EDF
   file and outputs a preictal-probability timeline.

---

## 3. Project structure

```
seizure-prediction/
├── data/                   # put downloaded .edf files here (per patient)
├── src/
│   ├── preprocess.py       # load EDF, filter, window, label
│   ├── features.py         # classical feature extraction
│   ├── dataset.py          # PyTorch Dataset for raw-signal deep model
│   ├── model.py            # CNN-LSTM architecture
│   ├── train_baseline.py   # RandomForest/XGBoost training
│   └── train_deep.py       # PyTorch training loop
├── requirements.txt
└── README.md
```

## 4. Setup
```bash
pip install -r requirements.txt
```

## 5. Run order
```bash
python src/preprocess.py --patient chb01 --data_dir data/chb01 --out_dir processed/chb01
python src/train_baseline.py --data_dir processed/chb01     # fast sanity check
python src/train_deep.py --data_dir processed/chb01         # main model
```

---

## Notes on doing this well (things papers get wrong / grading criteria care about)

- **No random shuffling before train/test split.** EEG windows near each other
  are highly correlated — random split leaks the answer and inflates accuracy.
  Split by time or by seizure event.
- **Class imbalance**: interictal data vastly outnumbers preictal. Use class
  weights or balanced sampling, not just accuracy as your metric (use AUC,
  F1, sensitivity/specificity instead).
- **Patient-specific vs patient-independent**: start patient-specific
  (train and test on the same patient, different seizures) — it's the easier
  and more standard first result. Cross-patient generalization is a great
  "future work" section but is a much harder, separate problem.

