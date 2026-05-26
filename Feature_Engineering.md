## Imputing Numerical Features:
When a dataset has missing values in numerical columns, you can't just leave them blank — most ML models will throw an error. Imputation is the process of filling in those missing values with a reasonable estimate.

### Common Strategies
### 1. Mean Imputation - Replace missing values with the average of the column
--> Best when data is normally distributed (no big outliers)
--> Simple and fast

example-
df['age'].fillna(df['age'].mean(), inplace=True)

### 2. Median Imputation
Replace with the middle value of the column.
--> Better than mean when there are outliers, since median is more robust

df['salary'].fillna(df['salary'].median(), inplace=True)

### 3. Mode Imputation
Replace with the most frequent value.
Rarely used for numerical data, but works for discrete numbers (e.g., number of children)

df['num_children'].fillna(df['num_children'].mode()[0], inplace=True)

### 4. Constant / Custom Value
Fill with a fixed value like 0 or -1, often to signal "unknown."

df['score'].fillna(0, inplace=True)

### 5. Model-Based Imputation (KNN, Regression)
Predict the missing value using other features. More accurate but computationally heavier.

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)

**Quick Rule of Thumb**

Situation --------------- Use

Clean, symmetric data-----Mean

Outliers present----------Median

Discrete numbers ---------Mode

Missing = meaningful -----Constant (e.g., 0 or -1)

High accuracy needed -----KNN / Model-based

**One Key Reminder**
Always fit the imputer on training data only, then transform both train and test — otherwise you leak information from the test set into training.

imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)  # ✅ No leakage
