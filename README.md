
# ML Classification with SMOTE and Different Algorithms

Welcome to this project repository!  
This project demonstrates how to handle imbalanced data using SMOTE (Synthetic Minority Oversampling Technique) and then apply several machine learning classifiers to perform predictions. The dataset used here contains two classes: **normal** and **Forwarding**. Since the "normal" class has many more samples than "Forwarding," SMOTE is used to balance the data before training models.

## What This Project Does

The dataset initially has a large imbalance between the two classes. This imbalance can lead to biased predictions. To fix this, SMOTE is applied to generate synthetic examples for the minority class. After balancing the dataset, several classification algorithms are implemented and evaluated:

- **Naïve Bayes:** A simple, fast, and probabilistic classifier that uses Bayes’ Theorem.
- **Decision Tree:** An interpretable classifier that splits the data based on feature values.
- **Random Forest:** An ensemble method that aggregates multiple decision trees for better accuracy.
- **Confusion Matrix:** An evaluation tool to see where predictions are correct and where errors occur.

## Repository File Structure

- **SMOTEdata.py**  
  Loads the CSV dataset, applies SMOTE to balance the classes, and saves the new, resampled dataset.  
  *Key Learning:* Preprocessing data and handling imbalanced classes.

- **naive_bayes.py**  
  Implements a Naïve Bayes classifier by splitting the data, training the model, and printing a classification report with metrics like precision, recall, and F1-score.  
  *Highlights:* Fast predictions with a probabilistic approach and basic evaluation.

- **confusion_matrix.py**  
  Focuses on computing and displaying a confusion matrix for the Naïve Bayes classifier. This file provides a clear view of true positives, false positives, true negatives, and false negatives.  
  *Benefits:* Helps in understanding specific model errors.

- **decision_tree.py**  
  Uses a Decision Tree classifier. The script splits the data, trains the model, and then outputs both a classification report and a confusion matrix.  
  *Benefits:* Offers an easy-to-understand view of the decision process and important features.

- **random_forest.py**  
  Implements a Random Forest classifier (an ensemble of decision trees). After splitting and training, it prints evaluation metrics and a confusion matrix.  
  *Benefits:* Provides improved accuracy and robustness compared to a single decision tree.

## How to Run the Code

### Prerequisites

Ensure that Python (version 3.13.1 or compatible) and the following packages are installed:

- pandas
- scikit-learn
- imbalanced-learn

Install the required packages with:
```bash
python3 -m pip install pandas scikit-learn imbalanced-learn
```

### Running the Scripts

1. **SMOTEdata.py**  
   Balances the dataset using SMOTE and saves a new CSV file.
   ```bash
   python3 SMOTEdata.py
   ```

2. **naive_bayes.py**  
   Trains a Naïve Bayes classifier and prints a classification report.
   ```bash
   python3 naive_bayes.py
   ```

3. **confusion_matrix.py**  
   Computes and displays the confusion matrix for the Naïve Bayes classifier.
   ```bash
   python3 confusion_matrix.py
   ```

4. **decision_tree.py**  
   Trains a Decision Tree classifier and outputs evaluation metrics.
   ```bash
   python3 decision_tree.py
   ```

5. **random_forest.py**  
   Trains a Random Forest classifier and prints out the evaluation metrics.
   ```bash
   python3 random_forest.py
   ```

## Project Motivation

This project is an effort to learn more about machine learning and how to handle imbalanced data effectively. The repository showcases different classification techniques and evaluation metrics to help understand model performance better. It is hoped that this work will be useful for others starting out with machine learning and facing similar challenges.

## Contributions

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit pull requests. Constructive feedback is highly appreciated!

## License

This project is licensed under the MIT License.
