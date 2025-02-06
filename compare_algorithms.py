# Import evaluation functions from the individual algorithm files.
# Make sure each file (naive_bayes.py, decision_tree.py, etc.) is in the same directory
# and that these functions are defined at the module level.

from naive_bayes import evaluate_naive_bayes
from decision_tree import evaluate_decision_tree
from random_forest import evaluate_random_forest
# from svm import evaluate_svm
from logistic_regression import evaluate_logistic_regression
from binary_classification import evaluate_binary_classification
from multiclass_classification import evaluate_multiclass_classification

def compare_all_algorithms():
    # File paths for the datasets
    original_dataset = 'datasetSMOTE.csv'
    resampled_dataset = 'resampled_dataset.csv'
    
    separator = "=" * 80

    # Naïve Bayes Comparison
    print(f"\n{separator}\nComparing Naïve Bayes:\n{separator}")
    evaluate_naive_bayes(original_dataset, "Original")
    evaluate_naive_bayes(resampled_dataset, "Resampled")

    # Decision Tree Comparison
    print(f"\n{separator}\nComparing Decision Tree:\n{separator}")
    evaluate_decision_tree(original_dataset, "Original")
    evaluate_decision_tree(resampled_dataset, "Resampled")

    # Random Forest Comparison
    print(f"\n{separator}\nComparing Random Forest:\n{separator}")
    evaluate_random_forest(original_dataset, "Original")
    evaluate_random_forest(resampled_dataset, "Resampled")

    # SVM Comparison
    # print(f"\n{separator}\nComparing SVM:\n{separator}")
    # evaluate_svm(original_dataset, "Original")
    # evaluate_svm(resampled_dataset, "Resampled")

    # Logistic Regression Comparison
    print(f"\n{separator}\nComparing Logistic Regression:\n{separator}")
    evaluate_logistic_regression(original_dataset, "Original")
    evaluate_logistic_regression(resampled_dataset, "Resampled")

    # Voting (Binary Classification) Comparison
    print(f"\n{separator}\nComparing Voting Classifier (Binary Classification):\n{separator}")
    evaluate_binary_classification(original_dataset, "Original")
    evaluate_binary_classification(resampled_dataset, "Resampled")

    # Multiclass Classification Comparison
    print(f"\n{separator}\nComparing Multiclass Classification:\n{separator}")
    evaluate_multiclass_classification(original_dataset, "Original")
    evaluate_multiclass_classification(resampled_dataset, "Resampled")

if __name__ == '__main__':
    compare_all_algorithms()
