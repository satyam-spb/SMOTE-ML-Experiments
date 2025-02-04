import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'datasetSMOTE.csv'
data = pd.read_csv(file_path)

# Check class distribution before SMOTE
print("Class distribution before SMOTE:")
print(data["Class"].value_counts())
'''
        Class distribution before SMOTE:
        Class
        normal        262851
        Forwarding      7645
        Name: count, dtype: int64
'''

# Separate  target(ie the "Class") and features(all columns other than "Class")
features = data.drop(columns=["Class"])
target = data["Class"]

# Ensure the target is in numeric form
target = target.astype("category").cat.codes  # Converts "normal" -> 0, "Forwarding" -> 1

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to balance the classes
features_resampled, target_resampled = smote.fit_resample(features, target)

# Convert resampled target back to original labels
target_resampled = pd.Series(target_resampled).map({0: "normal", 1: "Forwarding"})

# Combine resampled features and target into a new DataFrame
resampled_data = pd.concat([pd.DataFrame(features_resampled, columns=features.columns), 
                            target_resampled.rename("Class")], axis=1)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(resampled_data["Class"].value_counts())
'''
        Class distribution after SMOTE:
        Class
        Forwarding    262851
        normal        262851
        Name: count, dtype: int64
'''

# Save the resampled dataset
output_file = 'resampled_dataset.csv'
resampled_data.to_csv(output_file, index=False)
