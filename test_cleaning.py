import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load test.csv
test_df = pd.read_csv('test.csv')

print("The data columns present are: \n", test_df.columns)

print("The dataset looks like:\n", test_df.head(4))

print("The dataframe info is:\n")
test_df.info()

# Dropping 'Name' and 'Ticket' columns (similar to train.csv)
test_df1 = test_df.drop(columns=['Name', 'Ticket'])

# Dropping 'Cabin' column due to high percentage of missing values
test_df2 = test_df1.drop(columns=['Cabin'])

# Handle missing values in 'Embarked' column (if any)
test_df2.dropna(subset=['Embarked'], inplace=True)

# Copy the data to avoid chained assignment warnings
test_df3 = test_df2.copy()

# Imputing missing values in 'Age' with the mean
test_df3['Age'] = test_df3['Age'].fillna(test_df3['Age'].mean())

# Imputing missing values in 'Fare' (since test.csv has one missing value in 'Fare')
test_df3['Fare'] = test_df3['Fare'].fillna(test_df3['Fare'].mean())

# Checking for null values
print("Checking for null values again:\n")
print(round((test_df3.isnull().sum()/test_df3.shape[0]) * 100, 2))
print(test_df3.info())

# Scaling numerical columns using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
num_col_test = [col for col in test_df3.columns if test_df3[col].dtype != 'object']
x_test = test_df3.copy()
x_test[num_col_test] = scaler.fit_transform(x_test[num_col_test])

# Creating 'FamilySize' feature
x_test['FamilySize'] = x_test['SibSp'] + x_test['Parch'] + 1

# Dropping 'SibSp', 'Parch', and 'PassengerId' columns
x_test.drop(columns=['SibSp', 'Parch', 'PassengerId'], inplace=True)

# Save the cleaned test data to a CSV file
x_test.to_csv('cleaned_test.csv', index=False)

print("Test data has been cleaned and saved to 'cleaned_test.csv'.")
