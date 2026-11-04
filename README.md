# 📧 Email Spam Detection using Multinomial Naive Bayes

**Log transform + chi-square feature selection + threshold tuning on the UCI Spambase dataset**

B.Tech 2nd Year Machine Learning Mini Project (Topic 7: Email Spam Detection, Naive Bayes)

**Authors:** Parth Sidhapura (primary author), Vansh Patel, Riya Lakhani

---

## Overview

Naive Bayes is the classic spam filter because it is fast and easy to explain, but plain Multinomial Naive Bayes reaches only **78.9%** accuracy on the Spambase dataset because its features are very skewed.

We improve it with three simple steps, each chosen **using the training data only**:

1. **log(1 + x) transform** to shrink very large values so no feature dominates
2. **Chi-square feature selection** to keep the 20 most useful of the 57 features
3. **Threshold tuning** to move the spam cut-off from 0.5 to 0.6 (best cross-validated F1)

We also test **six engineered features** and compare against **seven baseline models**.

<p align="center">
  <img src="figures/fig00_pipeline.png" width="650" alt="Pipeline">
</p>

## Results

### Step-by-step improvement (test set, 842 emails)

| Step | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| 1. Multinomial NB (raw features) | 0.7886 | 0.7616 | 0.6845 | 0.7210 |
| 2. + log(1 + x) transform | 0.8848 | 0.7936 | **0.9613** | 0.8694 |
| 3. + chi-square selection (top 20) | 0.9026 | 0.8256 | 0.9583 | 0.8871 |
| 4. + tuned threshold (0.6) | **0.9169** | **0.8822** | 0.9137 | **0.8977** |

### Comparison with baseline models

| Model | Accuracy | Precision | Recall | F1-score | AUC-ROC | Train time (ms) |
|---|---|---|---|---|---|---|
| **Final improved MNB (ours)** | 0.9169 | 0.8822 | 0.9137 | 0.8977 | 0.9704 | ~3 |
| MNB + log | 0.8848 | 0.7936 | **0.9613** | 0.8694 | 0.9757 | ~3 |
| Gaussian NB | 0.8361 | 0.7240 | 0.9524 | 0.8226 | 0.9525 | ~5 |
| Bernoulli NB | 0.8943 | 0.8777 | 0.8542 | 0.8658 | 0.9609 | ~6 |
| Logistic Regression | 0.9347 | 0.9297 | 0.9048 | 0.9170 | 0.9768 | ~13 |
| Decision Tree (depth 5) | 0.8990 | 0.9169 | 0.8214 | 0.8666 | 0.9141 | ~21 |
| KNN (k = 5) | 0.9133 | 0.9148 | 0.8631 | 0.8882 | 0.9546 | **~2** |
| SVM (RBF) | 0.9323 | 0.9319 | 0.8958 | 0.9135 | 0.9772 | ~1064 |
| Random Forest (100 trees) | **0.9454** | **0.9367** | 0.9256 | **0.9311** | **0.9895** | ~463 |

Training times depend on your computer; all other numbers are exactly reproducible (`random_state=42`).

### Key findings

- The **log transform** is the most important step (+9.6 accuracy points).
- Using only **20 of 57 features** works *better* than using all of them.
- **Threshold tuning** halves the false positives (84 → 41 good emails wrongly marked as spam).
- The final model beats Bernoulli NB, Decision Tree and KNN and trains about **140× faster than Random Forest**.
- Only one of the six engineered features (`spam_score`) helped, because Naive Bayes already adds up the evidence of every word.
- Random Forest is still the most accurate model (94.5%), so the choice is a trade-off between accuracy and speed/simplicity.

<p align="center">
  <img src="figures/fig19_step_improvement.png" width="48%" alt="Step-by-step improvement">
  <img src="figures/fig18_threshold.png" width="48%" alt="Threshold tuning">
</p>
<p align="center">
  <img src="figures/fig12_all_confusion_matrices.png" width="90%" alt="Confusion matrices">
</p>

## Repository structure

```
email-spam-detection/
├── Email_Spam_Detection.ipynb   # main notebook with all code and outputs
├── spam_detection.py            # same code as a plain Python script
├── requirements.txt             # Python libraries needed
├── data/
│   ├── spambase.data            # UCI Spambase dataset
│   └── README.md                # dataset description
├── figures/                     # 25 plots produced by the code
├── results/
│   ├── results_comparison.csv          # all 8 base models
│   └── results_final_comparison.csv    # including the final improved model
├── paper/
│   └── IEEE_Email_Spam_Detection_Paper.docx   # research paper (IEEE format)
└── LICENSE
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/email-spam-detection.git
cd email-spam-detection

# 2. Install the libraries
pip install -r requirements.txt

# 3a. Open the notebook
jupyter notebook Email_Spam_Detection.ipynb

# 3b. Or run the script
python spam_detection.py
```

Run everything from the repository root folder. The code reads `data/spambase.data`, saves plots to `figures/` and tables to `results/`. A full run takes under a minute on a normal laptop (SVM cross-validation is the slowest part).

## What the code does

| Part | Details |
|---|---|
| Exploratory analysis | Class balance, key-word frequencies, correlation with label, heatmap, skewness before/after log |
| Preprocessing | Remove 391 duplicates, stratified 80:20 split, log(1 + x), standard scaling for LR/KNN/SVM |
| Feature engineering | `spam_score`, `ham_score`, `total_word_freq`, `total_char_freq`, `nonzero_count`, `capital_ratio` |
| Models | Multinomial, Gaussian and Bernoulli NB, Logistic Regression, Decision Tree, KNN, SVM, Random Forest |
| Evaluation | Accuracy, precision, recall, F1, AUC-ROC, average precision, confusion matrices, 5-fold CV, training time |
| Improvements | Chi-square feature selection (k chosen by CV), threshold tuning (chosen by CV) |
| Analysis | Ablation study, smoothing (alpha) study, learning curve, most important spam/ham words |

## Dataset

[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase): 4,601 emails, 57 numeric features (48 word frequencies, 6 character frequencies, 3 capital-letter statistics) and a spam/ham label. Collected at Hewlett-Packard Labs in 1999, licensed under CC BY 4.0. See [`data/README.md`](data/README.md).

## Main references

1. J. D. M. Rennie, L. Shih, J. Teevan, and D. R. Karger, "Tackling the poor assumptions of naive Bayes text classifiers," *ICML*, 2003.
2. Y. Yang and J. O. Pedersen, "A comparative study on feature selection in text categorization," *ICML*, 1997.
3. I. Androutsopoulos *et al.*, "An evaluation of naive Bayesian anti-spam filtering," *ECML Workshop*, 2000.
4. M. Hopkins, E. Reeber, G. Forman, and J. Suermondt, "Spambase," UCI Machine Learning Repository, 1999. https://doi.org/10.24432/C53G6X

The full reference list is in the paper.

## Authors

| Name | Role |
|---|---|
| **Parth Sidhapura** | Primary author |
| Vansh Patel | Co-author |
| Riya Lakhani | Co-author |

## License

Code: [MIT License](LICENSE). Dataset: CC BY 4.0 (UCI Machine Learning Repository).
