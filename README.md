
# SMOTE-ML-Experiments

Welcome to **SMOTE-ML-Experiments** – a beginner-friendly project that explores how to handle imbalanced data using SMOTE and compare various classification algorithms on both the original and resampled datasets.

The dataset used here has two classes: **normal** and **Forwarding**. Because the "normal" class is heavily overrepresented, SMOTE is applied to create a balanced dataset. In this project, models are trained on both the original dataset (`datasetSMOTE.csv`) and the SMOTE-resampled dataset (`resampled_dataset.csv`) so that the results can be compared.

## What’s in This Repo?

This repository includes multiple Python scripts that implement and evaluate different classifiers. All files (and datasets) are assumed to be in the same directory. The key files are:

- **SMOTEdata.py**  
  Loads the CSV dataset, applies SMOTE to balance the classes, and saves the resampled data as `resampled_dataset.csv`.  
  *Learning Focus:* Preprocessing and handling imbalanced classes with SMOTE.

- **naive_bayes.py**  
  Implements a Gaussian Naïve Bayes classifier. The script splits the data, trains the model, and prints a classification report for both the original and resampled datasets.  
  *Focus:* Fast probabilistic predictions and basic evaluation.

- **confusion_matrix.py**  
  Uses Naïve Bayes to generate and print confusion matrices for the original and resampled datasets, so you can see where predictions are correct and where they’re not.  
  *Focus:* Visualizing true/false positives and negatives.

- **decision_tree.py**  
  Implements a Decision Tree classifier. This file outputs both a classification report and a confusion matrix for the original and resampled data.  
  *Focus:* Interpretability and feature-based decision making.

- **random_forest.py**  
  Implements a Random Forest classifier (an ensemble of decision trees) and prints out the evaluation metrics (classification report and confusion matrix) for both datasets.  
  *Focus:* Improved accuracy and robustness with ensemble learning.

- **svm.py**  
  Implements a Support Vector Machine (SVM) classifier. This script evaluates the SVM model on both the original and resampled datasets.  
  *Focus:* Margin-based classification for binary tasks.

- **logistic_regression.py**  
  Uses Logistic Regression for binary classification. Evaluates the model with a classification report and confusion matrix on both datasets.  
  *Focus:* Probabilistic predictions via a linear model.

- **binary_classification.py**  
  Demonstrates a Voting Classifier that combines Logistic Regression and SVM for binary classification. The results are compared between the original and resampled data.  
  *Focus:* Combining models to potentially boost performance.

- **multiclass_classification.py**  
  Since the original dataset has only two classes, a synthetic multiclass scenario is created by splitting the "normal" class into "normal1" and "normal2". The updated script (which now fixes the mapping issue and avoids FutureWarnings) trains a multinomial Logistic Regression classifier and prints evaluation results for both the original and resampled datasets.  
  *Focus:* Extending binary classification to a multiclass problem and handling mapping properly.

## How to Run the Code

### Prerequisites

Make sure you have Python (3.13.1 or compatible) installed, along with the following packages:
- pandas
- scikit-learn
- imbalanced-learn

Install the required packages using:
```bash
python3 -m pip install pandas scikit-learn imbalanced-learn
```

### Running the Scripts

All scripts assume that both datasets are in the same directory as the code files.

1. **SMOTEdata.py**  
   Balances the dataset using SMOTE and saves the output as `resampled_dataset.csv`.
   ```bash
   python3 SMOTEdata.py
   ```

2. **naive_bayes.py**  
   Trains and evaluates a Naïve Bayes classifier on both the original and resampled datasets.
   ```bash
   python3 naive_bayes.py
   ```

3. **confusion_matrix.py**  
   Computes and prints the confusion matrix for Naïve Bayes on both datasets.
   ```bash
   python3 confusion_matrix.py
   ```

4. **decision_tree.py**  
   Trains a Decision Tree classifier and outputs evaluation metrics for both datasets.
   ```bash
   python3 decision_tree.py
   ```

5. **random_forest.py**  
   Trains a Random Forest classifier and prints evaluation results for both datasets.
   ```bash
   python3 random_forest.py
   ```

6. **svm.py**  
   Trains an SVM classifier and compares its performance on the original and resampled datasets.
   ```bash
   python3 svm.py
   ```

7. **logistic_regression.py**  
   Trains a Logistic Regression classifier for binary classification on both datasets.
   ```bash
   python3 logistic_regression.py
   ```

8. **binary_classification.py**  
   Demonstrates a Voting Classifier combining Logistic Regression and SVM for binary classification.
   ```bash
   python3 binary_classification.py
   ```

9. **multiclass_classification.py**  
   Creates a synthetic multiclass scenario by splitting "normal" into "normal1" and "normal2", and then trains a multinomial Logistic Regression classifier.  
   ```bash
   python3 multiclass_classification.py
   ```

## Project Motivation

This project was built to learn and explore:
- **Imbalanced Data Handling:** Using SMOTE to generate synthetic samples for a more balanced dataset.
- **Algorithm Comparison:** Implementing various classification algorithms (Naïve Bayes, Decision Trees, Random Forest, SVM, Logistic Regression) and comparing their outputs on both original and balanced datasets.
- **Evaluation Metrics:** Using confusion matrices and classification reports to understand model performance.
- **Multiclass Extension:** Experimenting with a synthetic multiclass scenario based on a binary dataset.

The aim is to share these experiments in a beginner-friendly way while showing responsibility and a willingness to learn and improve.

## Contributions

Contributions and suggestions are welcome. Feel free to fork this repository, submit pull requests, or share ideas for improvements. Constructive feedback is appreciated!

## License

This project is open-source under the MIT License.

---

Thank you for checking out the project. Happy coding and exploring machine learning experiments!  
  
--- 
