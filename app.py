from flask import Flask, request, jsonify, render_template
from disease_model import tree_to_code, check_pattern, getSeverityDict, getDescription, getprecautionDict, getInfo, text_to_speech
# Import pyttsx3 library
import pyttsx3
import numpy as np
import pandas as pd
import csv
import re
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
text_to_speech(text)
# dataset = pd.read_csv('Data/sy')

# Load your trained model if it is stored in an external file, otherwise define clf directly
# Example: clf = load_model('path_to_model.pkl')

# Assuming you are defining the model directly here, like in your notebook:
# Decision Tree Model Implementation
# Load Datasets for training and testing
training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')

# Number of rows and columns
shape = training.shape
print("Shape of Training dataset: ", shape)

# Description about dataset
description = training.describe()
description

# Information about Dataset
info_df = training.info()
info_df



# To find total number of null values in dataset
null_values_count = training.isnull().sum()
null_values_count

# Print First eight rows of the Dataset
training.head(8)

cols= training.columns
cols= cols[:-1]

# x stores every column data except the last one
x = training[cols]

# y stores the target variable for disease prediction
y = training['prognosis']

# Figsize used to define size of the figure
plt.figure(figsize=(10, 20))
# Countplot from seaborn on the target varable and data accesed from Training dataset
sns.countplot(y='prognosis', data=training)
# Tile for title of the figur
plt.title('Distribution of Target (Prognosis)')
# Show used to display the figure on screen
plt.show()

# Grouping Data by Prognosis and Finding Maximum Values
reduced_data = training.groupby(training['prognosis']).max()

# Display the first five rows of the reduced data
reduced_data.head()

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Features for testing except the last variable
testx    = testing[cols]

# Target variable for Testing
testy    = testing['prognosis']

# Transforming categorical value into numerical labels
testy    = le.transform(testy)

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
