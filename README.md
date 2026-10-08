# MNIST Digit Recognition with Ensemble Learning

Handwritten digit classifier built for CS-437 (Machine Learning) at WSUV, reaching **97.49% validation accuracy** using classical ensemble methods under a deliberate constraint: no neural networks.

## The problem

Classify handwritten digits 0 through 9 from raw pixel data. The MNIST dataset consists of 28x28 grayscale images, each represented as 784 pixel intensity values.

The project requirement was to solve this without reaching for deep learning, so the challenge was getting competitive accuracy out of classical models through preprocessing and ensembling rather than model capacity.

## Approach

The whole thing runs as a single scikit-learn `Pipeline` with three stages, which keeps preprocessing and modeling in one object and prevents data leakage during cross-validation.

**1. Normalization.** Pixel values are scaled from 0–255 down to [0, 1] with a `FunctionTransformer`. Logistic regression in particular converges much better on scaled inputs.

**2. Dimensionality reduction.** PCA reduces the 784 raw pixel features to 50 components. Most of the variance in MNIST lives in a small number of components, and cutting the feature space by more than 90% made the grid search tractable without meaningfully hurting accuracy.

**3. Ensemble classification.** A `VotingClassifier` combines two models with different inductive biases:

- **Logistic Regression** (`max_iter=1000`), which draws linear decision boundaries in PCA space
- **Random Forest** (`n_estimators=200`), which captures non-linear feature interactions

Soft voting averages the two models' predicted probabilities rather than taking a majority vote, so a confident model outweighs an uncertain one. The two make different kinds of mistakes, so combining them beat either one alone.

**Hyperparameter tuning.** `GridSearchCV` with 3-fold cross-validation searched over the regularization strength `C` for logistic regression (0.01, 0.1, 1) and the relative weighting between the two voters, scoring on accuracy.

## Results

| Metric | Score |
|---|---|
| Validation accuracy | **97.49%** |
| Kaggle test score | **0.9411** |

The Kaggle score corresponds to roughly 39,500 of 42,000 test digits classified correctly.

The notebook also plots predictions against true labels for a sample of test images, which made the failure cases easy to inspect. Most misclassifications were digits that are genuinely ambiguous when written quickly.

## Tech stack

- **Python**
- **scikit-learn** — `Pipeline`, `PCA`, `VotingClassifier`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `FunctionTransformer`
- **pandas / NumPy** — data loading and array handling
- **Matplotlib** — visualizing digits and prediction results
- **Jupyter Notebook**

## Repository contents

| File | Description |
|---|---|
| `Term_Project.ipynb` | Full analysis: data loading, preprocessing, model training, tuning, and evaluation |
| `submission.csv` | Final predictions submitted to the Kaggle competition |

## Data

The dataset is not included in this repository. It comes from the
[Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer),
where `train.csv` and `test.csv` can be downloaded directly. Place them beside the notebook to run it.
