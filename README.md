# Email Spam Filter — Supervised Machine Learning with Linear Regression

A **beginner-friendly, runnable Python project** that demonstrates how to train, test, and use a spam detector with a **labeled email dataset**. It uses **ordinary linear regression** deliberately to teach the foundations of supervised learning. Although the result is a binary decision, this is **not the best algorithm for a production spam classifier**; see [Why linear regression?](#why-linear-regression) for the important distinction.

**Learning objectives**

1. Identify input features (`subject`, `sender`, `body`) and the target label (`spam` or `ham`).
2. Convert raw text into numeric features using **TF-IDF**.
3. Divide the dataset into **80% training / 20% testing**, preserving the proportion of each label.
4. Fit a linear regression model on training examples *only*.
5. Convert a continuous numeric output into a spam/ham decision using a **0.5 threshold**.
6. Evaluate the model on unseen test examples using accuracy, precision, recall, F1 and a confusion matrix.
7. Save the trained pipeline and classify a new email from the command line.

> **Educational disclaimer:** This dataset is **synthetic** (100 invented emails, no real personal information). It was constructed to illustrate spam-related patterns and is **much easier than real email**. A strong test score on this data does **not** prove the system is accurate or safe for use with real mail. It does not connect to an inbox, read passwords, click links or send messages.

---

## 1. Repository structure

```text
email-spam-linear-regression/
├── README.md                       # Explanations, instructions, and exercises
├── LICENSE                         # MIT license
├── .gitignore                      # Ignore virtualenvs, Python caches, and saved models
├── requirements.txt                # Python packages
├── data/
│   └── emails.csv                  # 100 labeled example emails
├── spam_filter/
│   ├── __init__.py                 # Make this folder a Python package
│   ├── train.py                    # Load, split, train, test, save
│   └── predict.py                  # Classify a new email
└── tests/
    └── test_spam_filter.py         # Automated checks
```

A folder named `artifacts/` will be created automatically after training. It holds the saved model and is deliberately excluded from Git, because it is a generated file.

## 2. What is the dataset?

`data/emails.csv` has **100 rows**: **50 spam** and **50 ham (legitimate email)**. Every example has four columns:

| Column | Input or target? | Meaning | Example |
|---|---|---|---|
| `subject` | Input (feature) | Email subject | `URGENT: Verify your password` |
| `sender` | Input (feature) | Sender's email address / domain | `security-alert@account-check.example` |
| `body` | Input (feature) | Email message text | `Click now to confirm your password...` |
| `label` | Target | Correct answer from the labeled dataset | `spam` or `ham` |

The target is encoded numerically by the training code as follows:

- `ham` → **0** (not spam)
- `spam` → **1** (spam)

**What might indicate spam?** Phrases such as `urgent`, `password`, `click now`, `verify your account`, `claim a prize`, `provide banking details`, and `pay a fee` may be useful clues. The program does **not** simply implement a hard-coded blacklist of these words: it learns weights for words and pairs of words from labeled examples. Sender text is included as a separate feature group.

**Important:** A legitimate email can contain `urgent`, `password`, or `click now`. The provided dataset deliberately includes some of these words in normal messages. The words alone do not prove an email is malicious. Likewise, a believable sender address is not proof of authenticity: the header can be spoofed. All addresses use the reserved `.example` domain.

## 3. Requirements and setup

- Python **3.10 or later** is recommended.
- A terminal (Command Prompt, PowerShell, VS Code terminal, macOS Terminal, or Linux shell).
- Internet access once to install the Python packages. Training itself is offline.

**Windows PowerShell** (run in the project directory):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can use `.venv\Scripts\python.exe` instead of activating, or use Command Prompt with `.venv\Scripts\activate.bat`.

**macOS / Linux** (run in the project directory):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Packages in `requirements.txt`**

| Library | Role in this example |
|---|---|
| `pandas` | Read the CSV into a DataFrame and choose feature / label columns. |
| `scikit-learn` (`sklearn`) | Split data, vectorize text, combine features, fit linear regression, and calculate evaluation metrics. |
| `numpy` | Convert continuous predictions into binary integers with a 0.5 threshold. |
| `joblib` | Save and reload the trained text-processing + regression pipeline. |
| `pytest` | Run automated checks on the dataset and learning workflow. |

The standard-library modules `argparse` (command-line arguments) and `pathlib` (portable file paths) are bundled with Python and do not require installation. **Never load a `.joblib` file from an untrusted source**: model files use pickle-like deserialization and may execute code when loaded.

## 4. Train the model and evaluate it (80/20)

From the project root, run:

```bash
python -m spam_filter.train
```

The flow is:

```text
100 labeled emails (50 spam + 50 ham)
           |
           v
Check data columns, missing values, labels, duplicates
           |
           v
Stratified random split (seed = 42)
           |
      +----+------------------+
      |                       |
      v                       v
TRAIN: 80 emails          TEST: 20 emails
40 spam / 40 ham          10 spam / 10 ham
      |                       |
      v                       |  untouched during training
Fit TF-IDF on TRAIN only       |
      |                       |
      v                       |
Fit LinearRegression          |
      |                       |
      +-----------+-----------+
                  |
                  v
     Transform held-out TEST emails
                  |
                  v
   Predict raw numerical scores
                  |
                  v
  If score >= 0.5: spam, else ham
                  |
                  v
  Compare predictions with test labels
                  |
                  v
   Print metrics and save the model
```

The actual split is performed by:

```python
train_test_split(
    features,
    targets,
    test_size=0.20,
    random_state=42,
    stratify=targets,
)
```

- `test_size=0.20`: hold out 20% for testing, so the other 80% is for training.
- `random_state=42`: reproducible random shuffling (the seed is an arbitrary constant).
- `stratify=targets`: preserve the spam/ham proportion in both subsets.
- **No leakage:** TF-IDF learns its vocabulary and term statistics *after the split*, on the training subset only. The held-out test examples are not used to build the model.

### A sample verified run

With the bundled data and the stated environment, the implementation produced:

```text
Loaded dataset: 100 emails (50 spam, 50 ham)
Training: 80 emails | Testing: 20 emails
Training classes: {0: 40, 1: 40}
Testing classes:  {0: 10, 1: 10}
accuracy: 1.000
precision_spam: 1.000
recall_spam: 1.000
f1_spam: 1.000
mean_squared_error: 0.065
Confusion matrix (TN, FP, FN, TP): {'TN': 10, 'FP': 0, 'FN': 0, 'TP': 10}
```

**Interpret this carefully.** The score is perfect on these **20 easy synthetic test examples**. This is a tiny, artificial and potentially unrepresentative benchmark; the model may fail on unseen writing styles, sender domains, phishing techniques or real legitimate messages. It should not be used to filter real inboxes. Results may differ with another dataset or library version.

You can choose different paths or try a different threshold:

```bash
python -m spam_filter.train --data data/emails.csv --output artifacts/my_model.joblib --threshold 0.6
```

## 5. How text becomes numbers: TF-IDF

A mathematical model cannot work directly with strings such as `Click now to confirm your password`.

We use **term frequency–inverse document frequency** (`TfidfVectorizer`) to convert each text column into numbers:

1. **Tokenization:** Split text into terms (for example `click`, `password`, `urgent`).
2. **N-grams:** Optionally include adjacent word pairs (for example `click now` or `verify password`).
3. **Term frequency:** Capture the presence or frequency of a term in the email field.
4. **Inverse document frequency:** Downweight words seen across many examples; distinguish more informative terms.
5. **Feature matrix:** Build a numeric row for every email and a feature column for each learned term.

`ColumnTransformer` creates **three separate vectorized groups** so the model can learn distinct weights for subject, sender and body:

| Group | Processing | Why? |
|---|---|---|
| `subject` | TF-IDF, 1-word and 2-word phrases | Short headline patterns like `urgent` or `verify account`. |
| `sender` | TF-IDF with sender-friendly tokenization | Local parts and domains can carry patterns, but are not trustworthy identity checks. |
| `body` | TF-IDF, 1-word and 2-word phrases | Learn message content such as `click now` and `send password`. |

The model may learn that certain terms correlate with spam in its **training set**. It does **not** understand the intent of a message or verify the sender's identity.

## 6. Why linear regression?

### What it normally does

**Linear regression** predicts a *continuous numeric value* from features. It learns an intercept and feature weights, minimizing **ordinary least squares (OLS)** error:

\[
\widehat{y} = b_0 + \sum_{j=1}^{p} w_j x_j
\]

\[
\text{minimize } \sum_{i=1}^{n} (y_i - \widehat{y}_i)^2
\]

Where:

- `x_j` = a numerical TF-IDF feature, such as the weight of `urgent` or `click now`;
- `w_j` = a learned coefficient for that feature;
- `b_0` = the intercept;
- `y_i` = the known label (0 = ham, 1 = spam);
- `ŷ_i` = the model's continuous prediction.

### How we adapt it to spam classification

The *true labels* are binary (0/1), but ordinary linear regression still produces continuous scores. We manually turn them into classifications:

| Raw score | Decision with threshold 0.5 |
|---:|---|
| `0.18` | ham (0) |
| `0.49` | ham (0) |
| `0.50` | spam (1) |
| `0.83` | spam (1) |
| `1.12` | spam (1), **not** a 112% probability |
| `-0.16` | ham (0), **not** a negative probability |

**Why teach it anyway?** It makes the end-to-end ML process particularly transparent: labeled examples → numerical features → learned coefficients → prediction → threshold → evaluation. It also exposes an important modeling lesson: **an algorithm can be made to produce a classification without being a good choice for classification**.

### Why logistic regression is normally a better choice

| Linear regression (this educational example) | Logistic regression (recommended starting point) |
|---|---|
| Built for predicting continuous numbers. | Built for classifying binary outcomes. |
| Uses squared-error loss. | Uses a binary classification loss (log loss). |
| Can output below 0 or above 1. | Outputs scores between 0 and 1, interpreted as estimated class probabilities under its modeling assumptions (not automatically well calibrated). |
| Requires an external classification threshold. | Also uses a threshold for the final class decision. |
| No built-in regularization in `LinearRegression`. | `LogisticRegression` in scikit-learn uses regularization by default. |

**Do not confuse the names:** logistic regression is a **classification** method, despite including the word “regression.” For a genuine spam filter, start with `LogisticRegression`, `MultinomialNB`, or another classifier, and validate on representative real-world data with appropriate privacy and security controls.

## 7. Understand model evaluation

`evaluate_model` computes performance **only on the test set**:

| Metric | What it answers |
|---|---|
| **Accuracy** | What fraction of all test emails was classified correctly? |
| **Precision (spam)** | Of emails flagged as spam, how many really were spam? |
| **Recall (spam)** | Of all real spam emails, how many did we catch? |
| **F1 (spam)** | Harmonic mean balancing precision and recall. |
| **MSE** | Average squared difference between true 0/1 labels and raw regression scores. |

Confusion matrix interpretation:

| Actual \ Predicted | Ham (0) | Spam (1) |
|---|---|---|
| **Ham (0)** | **TN** — correctly kept | **FP** — legitimate mail wrongly flagged |
| **Spam (1)** | **FN** — spam missed | **TP** — correctly caught |

For a real inbox, **false positives** can be especially costly because important legitimate messages might disappear into a spam folder. Precision and recall are often more informative than accuracy alone, especially if there are many more ham messages than spam.

**Threshold effect:** Raising the threshold to 0.6 generally makes the filter less likely to mark messages as spam (often fewer false positives but more misses). Lowering it generally increases spam detection but can increase false positives. The exact effect depends on observed scores, and the threshold must be chosen using validation data, **not tuned repeatedly on the final test set**.

## 8. Predict whether a new email is spam

First run the training command; it creates `artifacts/spam_linear_regression.joblib`. This file includes both the trained text vectorizers and the fitted linear regression model, so prediction uses *exactly the same feature transformations* learned from the training data.

**Example: suspicious message**

```bash
python -m spam_filter.predict \
  --sender "alert@account-check.example" \
  --subject "URGENT: Verify your password" \
  --body "Click now to confirm your password and claim your prize"
```

On **Windows PowerShell**, enter the same command on a single line (the backslash shown above is only a multiline marker for Unix shells):

```powershell
python -m spam_filter.predict --sender "alert@account-check.example" --subject "URGENT: Verify your password" --body "Click now to confirm your password and claim your prize"
```

**Example: legitimate message**

```bash
python -m spam_filter.predict --sender "lecturer@university.example" --subject "Course materials" --body "The lecture notes and lab exercises are now available in the class workspace"
```

You will see a predicted label (`SPAM` or `HAM`) and a raw regression score. **The score is not a percentage and not a calibrated probability.** The model cannot guarantee that either example is correctly labeled.

Use a different saved model path with `--model artifacts/my_model.joblib`.

## 9. Python functions: what every step does

### `spam_filter/train.py`

| Function | Inputs | Output / purpose |
|---|---|---|
| `load_emails(csv_path)` | Path to labeled CSV | Returns a `pandas.DataFrame`, validating columns, missing values, duplicate messages and allowed labels. |
| `split_emails(df, test_size=0.20, random_state=42)` | DataFrame, split fraction, seed | Returns `X_train`, `X_test`, `y_train`, `y_test`, stratified by label. |
| `build_model()` | None | Returns a scikit-learn `Pipeline` containing `ColumnTransformer` (3 TF-IDF vectorizers) followed by `LinearRegression`. |
| `classify_scores(scores, threshold=0.5)` | Continuous predictions and threshold | Returns a NumPy array of `0` (ham) or `1` (spam). |
| `evaluate_model(model, x_test, y_test, threshold=0.5)` | Trained model and held-out examples | Returns a dictionary of MSE, accuracy, spam precision/recall/F1, confusion-matrix counts and a classification report. |
| `train_and_evaluate(csv_path, model_path, threshold)` | Dataset path, save path, threshold | Executes the full workflow, prints results, saves the trained model with `joblib`, and returns test metrics. |
| `parse_args()` | Command-line options | Reads `--data`, `--output`, and `--threshold`. |
| `main()` | None | CLI entry point calling `train_and_evaluate`. |

### `spam_filter/predict.py`

| Function | Inputs | Output / purpose |
|---|---|---|
| `predict_email(model_path, subject, sender, body)` | Trusted trained-model path and three email fields | Returns `(label, raw_score)` for a single email. |
| `parse_args()` | Command-line options | Reads `--sender`, `--subject`, `--body`, and optional `--model`. |
| `main()` | None | Displays the final label and raw score. |

### Important methods from the libraries

| Method / class | Role |
|---|---|
| `pd.read_csv(...)` | Loads the dataset. |
| `train_test_split(...)` | Randomly assigns examples to training and test sets. |
| `TfidfVectorizer(...)` | Learns word/phrase features and converts text into numbers. |
| `ColumnTransformer(...)` | Applies a different text vectorizer to each selected column. |
| `Pipeline(...)` | Chains feature extraction and the estimator together. |
| `LinearRegression()` | Creates the OLS regression estimator. |
| `model.fit(X_train, y_train)` | Learns vocabulary/TF-IDF statistics and regression coefficients using training data. |
| `model.predict(X_test)` | Returns continuous scores for previously held-out messages. |
| `confusion_matrix(...)` | Counts correct and incorrect classification types. |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Measure binary classification performance. |
| `mean_squared_error(...)` | Measures the squared error of continuous regression scores. |
| `classification_report(...)` | Shows per-class precision, recall, F1 and support. |
| `joblib.dump(...)`, `joblib.load(...)` | Persist and restore the trained pipeline. |

## 10. Run automated tests

```bash
python -m pytest -q
```

The tests verify:

1. There are 100 examples and both labels occur 50 times.
2. The 80/20 split is exactly 80 and 20, with class counts 40/40 and 10/10.
3. The same email rows do not appear in both subsets.
4. The 0.5 threshold correctly handles scores outside the 0–1 range.
5. The training and test pipeline runs end-to-end, produces metrics and saves a model.
6. Missing required fields cause a clear validation error.

The initial implementation was verified with **6 passing tests**. We intentionally do **not** assert perfect accuracy in tests: a real ML pipeline should remain valid even when some unseen examples are misclassified.

## 11. Student exercises

1. **Inspect the dataset:** Find three spam subjects and three normal subjects. Do they share any words?
2. **Change the split:** Change `test_size` from `0.20` to `0.30`. How many messages become training vs test samples? Does the score change?
3. **Experiment with keywords:** Add new labeled rows with both malicious and harmless uses of `urgent`, `password`, and `click now`. Keep both classes represented. Re-train and observe any difference.
4. **Change the threshold:** Train with `--threshold 0.4` and again with `--threshold 0.6`. Compare the confusion matrices; discuss false positives and false negatives.
5. **Compare algorithms:** Replace `LinearRegression()` with `LogisticRegression(max_iter=1000)` and adapt prediction to use either `predict_proba(... )[:, 1]` or binary class predictions. Compare F1 and discuss why a classifier is more appropriate.
6. **Improve validation:** Evaluate using cross-validation on development data, then preserve a final untouched holdout set for one final assessment. Do not optimize based on the final test results.

**Discussion question:** Why can a model show 100% accuracy on 20 synthetic test emails yet still be unsafe for real email filtering? Consider dataset size, realism, class balance, sender spoofing, domain shifts and adversarial wording.

## 12. Limitations and responsible use

- **Synthetic data:** Results are not representative of production inboxes. This is a toy supervised-learning demonstration.
- **Small sample:** 80 training messages provide weak evidence of generalization, especially with many text features.
- **Linear regression:** Ordinary least squares lacks probability constraints and is sensitive to high-dimensional and correlated features; a purpose-built classifier is recommended.
- **Sender ≠ verified identity:** The text of a sender address can be forged; real systems also consider SPF/DKIM/DMARC checks, sender reputation, headers and URLs.
- **No inbox integration:** This program never connects to Gmail or Outlook, never clicks any URLs, and never sends or deletes emails.
- **Bias and drift:** Marketing emails, unusual legitimate requests, languages, domains and attackers' tactics evolve. Test against relevant, permissioned, representative data and monitor mistakes.
- **Privacy:** Avoid putting real email addresses, secrets, private messages or personal data in a public GitHub repository.
- **Model files:** Never load untrusted `.joblib` files. Only load models you created in a trusted environment.

## 13. Publish this folder as a new GitHub repository

The following commands assume you have created an **empty** GitHub repository called `email-spam-linear-regression` in your account at <https://github.com/new>. Do not initialize it with another README or license because those files already exist here.

From inside this folder:

```bash
git init -b main
git add .
git commit -m "Add educational spam filter using linear regression"
git remote add origin https://github.com/YOUR_USERNAME/email-spam-linear-regression.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username. You may need to authenticate using Git Credential Manager or GitHub CLI (a personal access token may be required when using HTTPS). Do **not** commit `.venv/`, `artifacts/` or real private emails.

## Further reading (official documentation)

- [scikit-learn: Linear models, including OLS and logistic regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [scikit-learn: classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Suggested class activity:** Have students predict the label of five example emails *before* running the code. Then inspect each real label, compare predictions and discuss false positives, false negatives, and why a high score on a synthetic dataset can be misleading.
