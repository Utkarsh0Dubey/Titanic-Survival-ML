import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('train.csv')

print("The data columns present are: \n", df.columns)

print("The dataset looks like:\n", df.head(4))

print("The dataframe info is:\n")
df.info()

"""
The output of above line is:
Missing Data: The columns with missing data are Age (86 missing values), Fare (1 missing value), and 
Cabin (327 missing values).
The function displays the no of non-null values. To obtain no of missing values, we subtract from 418. 
Data Types: The data types in your DataFrame are a mix of integers, floating-point numbers, and objects 
(strings or mixed data).
"""

print("Checking for duplicate rows:")
count = 0
# df.duplicated returns a boolean series
for is_duplicate in df.duplicated():
    if is_duplicate:
        count += 1
print("Duplicate count is:", count)  # prints 0

# Categorical Columns
cat_col = [col for col in df.columns if df[col].dtype == object]

print("The categorical Columns are:")
print(cat_col)
# Finding the no of unique values in Categorical Values:
print("No of unique values in each column are:")
print(df[cat_col].nunique())
print(df.shape)

# Dropping names and tickets to get a new dataframe
df1 = df.drop(columns=['Name', 'Ticket'])
print(df1.shape)

"""
"Name" and "Ticket" columns are being dropped in this because
 they are deemed to have minimal influence on the 
 target variable without additional feature engineering.
"""

"""
Output of above 2 lines: (x, y) means dataframe has x rows and y columns.
(418, 11)
(418, 9)
"""

# Finding the % of missing values in each column: (True = 1)
print(round((df1.isnull().sum()/df1.shape[0]) * 100, 2))
"""
The entire expression calculates the 
percentage of missing values for each column in the DataFrame df1.
It then rounds those percentages to two decimal places.
The result is a pandas Series where the index represents the 
column names, and the values represent the percentage of missing
data in each column.
"""

# - Drop the "Cabin" Column
# - The "Cabin" column has 77% missing values, which is too high to fill reliably.
# - Dropping this column to avoid introducing bias or noise into the dataset.
df2 = df1.drop(columns=['Cabin'])

# - Handle Missing Values in the "Embarked" Column
# - The "Embarked" column has only 0.22% missing values.
# - Even though this is a small percentage, it's a good practice to handle the missing data
#   to ensure data completeness and avoid potential issues with certain machine learning models.
# - Since the percentage is so low, dropping the rows with missing
# - "Embarked" values will have minimal impact on the dataset.
df2.dropna(subset=['Embarked'], inplace=True)

# Step 4: Decide What to Do with the "Age" Column
# - The "Age" column has 19.87% missing values. This is significant, but not so high that we should drop the column.
# - Instead of dropping it, we'll consider imputation strategies
# - (mean, median, or other methods) to fill in the missing values.
# - For now, we're not dropping the "Age" column,
# - as it's likely to be an important feature for analysis.
# - We will use imputation strategies for age.

print(df1.shape)
print(df2.shape)

"""
Types of imputing: Mean and Median:
Mean is used when data has no extreme outliers and data is normally distributed.
Median is used when data has outliers or is skewed.
"""

# Mean Imputing the age column:
df3 = df2.fillna(df2.Age.mean())

"""
These 2 lines in the general case are better for imputation. 
This is because this will change only null values in age column. But, in this
case only age column as null value so it will work.
df3 = df2.copy()  # Create a copy of df2 to avoid modifying it directly
df3['Age'].fillna(df3['Age'].mean(), inplace=True)  # Mean imputation for the 'Age' column
"""

print("Checking for null values again:\n")
print(round((df3.isnull().sum()/df3.shape[0]) * 100, 2))
print(df3.info())

# By any of the 2 above lines, we can see that no null values remain.
"""
# We create the box plot of 'Age' data.
plt.boxplot(df3['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()


# The box plot shows that 50% of people are aged between approximately 22 and 34.
# The data is relatively balanced with no extreme skewness, but there are outliers:
# some individuals are younger than 5 or older than 55.
"""

# Calculating the summary statistics, so we can drop the outliers.
mean = df3['Age'].mean()
std = df3['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print("Lower Bound = ", lower_bound)
print("Upper Bound = ", upper_bound)

# Drop the outliers:
df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]

# Scaling

# initialising the MinMaxScaler
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
scaler = MinMaxScaler(feature_range=(0, 1))

# Numerical columns
num_col_ = [col for col in df4.columns if df4[col].dtype != 'object']
x1 = df4
# learning the statistical parameters for each of the data and transforming
x1[num_col_] = scaler.fit_transform(x1[num_col_])
print(x1.head(5))
x1['FamilySize'] = x1['SibSp'] + x1['Parch'] + 1
x1.drop(columns=['SibSp', 'Parch', 'PassengerId'], inplace=True)
print(x1.columns)

# Removing the remaining outliers: Looking at the columns, I think more removal would lead to over-fitting.

# Saving the csv file:
x2 = x1.copy()
x2.to_csv('cleaned_titanic.csv', index=False)

# Count the total number of missing values in the entire DataFrame
total_missing = x2.isnull().sum().sum()
print(f"Total missing values: {total_missing}")

