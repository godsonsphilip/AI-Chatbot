from flask import Flask, request, jsonify, render_template
from disease_model import tree_to_code, check_pattern, getSeverityDict, getDescription, getprecautionDict, getInfo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib  # Optional for saving/loading models


app = Flask(__name__)
dataset = pd.read_csv('path_to_your_dataset\\\.csv')

# Load your trained model if it is stored in an external file, otherwise define clf directly
# Example: clf = load_model('path_to_model.pkl')

# Assuming you are defining the model directly here, like in your notebook:
# Decision Tree Model Implementation
clf1  = DecisionTreeClassifier()

# Fitting the Training Data
clf = clf1.fit(x_train,y_train)

# Cross-Validation for Model Evaluation
scores = cross_val_score(clf, x_test, y_test, cv=3)

# Print the Mean Score
print("Mean Score: ",scores.mean()) 

# Creating Support Vector Machine Model
model=SVC()

# Train the model on Training Data
model.fit(x_train,y_train)

# Print accuracy for SVM Model on the training set
print("Accuracy score for svm: ", model.score(x_test,y_test))

# Calculate feature importance using the trained Decision tree classifier
importances = clf.feature_importances_

# Sort indices in descending order based on feature importance
indices = np.argsort(importances)[::-1]

# Get feature names corresponding to their importance score
features = cols

 # Make sure to train this model with your dataset
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
# Train the model here or load trained parameters, then pass it to tree_to_code

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file to take user input

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['symptom']  # Fetch the symptom from the HTML form
    print(f"User symptom input: {data}")
    
    # Here you would use `tree_to_code` or another function that processes `data`
    # and interacts with your trained classifier to make a prediction.
    
    # Simulate the interaction using example functions (replace with your actual function call)
    # Ensure you capture the output in the way it is printed to the user
    try:
        prediction = tree_to_code(clf, data)  # Pass the input data to the function
        return jsonify(prediction=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
