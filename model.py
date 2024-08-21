import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the cleaned training data
df = pd.read_csv('cleaned_titanic.csv')

# Step 1: One-hot encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 2: Separate features and target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

# Step 3: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions on the validation set
y_pred = model.predict(X_val)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Load the cleaned test data
test_df = pd.read_csv('cleaned_test.csv')

# Step 7: Apply the same one-hot encoding to the test set
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

# Align test set columns with training set columns
test_X = test_df.reindex(columns=X.columns, fill_value=0)  # Ensure test data has the same columns as training data

# Make predictions on the test set
test_predictions = model.predict(test_X).astype(int)

# Load the original test.csv to retrieve PassengerId for the submission
original_test_df = pd.read_csv('test.csv')

# Prepare the submission file with integer predictions
submission = pd.DataFrame({
    'PassengerId': original_test_df['PassengerId'],
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('titanic_submission_rf.csv', index=False)
print("Submission file 'titanic_submission_rf.csv' has been created.")
