import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tabulate import tabulate

# Load dataset
file_path = 'adult.data'
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 
    'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
    'native-country', 'income'
]

data = pd.read_csv(file_path, names=columns, na_values=' ?', skipinitialspace=True)

# Drops the rows that have missing values
data.dropna(inplace=True)

# Encodes the variables
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Splits the dataset into features
X = data.drop('income', axis=1)
y = data['income']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to display results in table format
def display_results(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    # Converts the classification report to DataFrame and format it
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
    
    # Prints the results using tabulate for better readability
    print(f"\n--- {model_name} Results ---")
    print(tabulate(report_df, headers='keys', tablefmt='grid'))
    print("\nConfusion Matrix:")
    print(tabulate(matrix, headers=['Predicted 0', 'Predicted 1'], tablefmt='grid'))
    print(f"\nOverall Accuracy: {accuracy:.2%}\n")

# Function to train and evaluate Logistic Regression
def logistic_regression():
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_results("Logistic Regression", y_test, y_pred)

# Function to train and evaluate the Support Vector Machine
def support_vector_machine():
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    display_results("Support Vector Machine (SVM)", y_test, y_pred)

# Asks the user what model they want to use
def main():
    print("Choose an option:")
    print("1 - Logistic Regression")
    print("2 - Support Vector Machine")
    print("3 - Both")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        logistic_regression()
    elif choice == '2':
        support_vector_machine()
    elif choice == '3':
        logistic_regression()
        support_vector_machine()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
