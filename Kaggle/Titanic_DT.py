

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle

train_data = pd.read_csv("Kaggle_Contest_main\\training.csv")
test_data = pd.read_csv("Kaggle_Contest_main\\testing.csv")

label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

# Selecting the relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train_data[features]
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Training 
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

def calculate_accuracy():
    return float(accuracy)


input_passenger = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.5
}

# Convert input_passenger dictionary to a DataFrame
input_df = pd.DataFrame([input_passenger])

# Encode categorical variables using LabelEncoder
input_df['Sex'] = label_encoder.transform(input_df['Sex'])

# Selecting the relevant features
X_input = input_df[features]

# Making prediction for the single entry
single_prediction = dt_model.predict(X_input)

# Outputting prediction
pp=int(single_prediction[0])
print(type(pp))




"""
model_filename = 'saved_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(dt_model, file)

  """