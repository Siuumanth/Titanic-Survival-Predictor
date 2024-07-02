# Attempting to solve the titanic problem using Decision Tree Model in Python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

# Selecting the relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train_data[features]
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Making predictions on the validation set
y_pred = dt_model.predict(X_val)

# Evaluating the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Classification Report: \n{class_report}')

# Making predictions on the test set
X_test = test_data[features]
test_predictions = dt_model.predict(X_test)

# Preparing submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission_decision_tree.csv', index=False)


# Confusion Matrix results
# TrueNegative(TN)    FalsePositive(FP)
# FalseNegative(FN)  TruePositive(TP)


# Assuming 'input_passenger' is a dictionary containing relevant information about the passenger
input_passenger = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.5
}

# Encoding categorical variables
input_passenger['Sex'] = label_encoder.transform([input_passenger['Sex']])[0]

# Creating feature array for prediction
input_features = [input_passenger[feature] for feature in features]

# Making prediction
prediction = dt_model.predict([input_features])

# Outputting prediction
if prediction == 1:
    print("The passenger is predicted to have survived.")
else:
    print("The passenger is predicted to have not survived.")