import re
import csv
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pyttsx3


engine = pyttsx3.init()

severityDictionary = {}
description_list = {}  # Make sure this is defined
precautionDictionary = {}

# Add all the necessary functions you used for processing the data and making predictions


def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
pass

def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))
pass

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])
pass

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello", name)
pass




def getprecautionDict():
    global precautionDictionary
    with open('Data/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)
pass

def getSeverityDict():
    global severityDictionary
    with open('Data/Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
pass

def getDescription():
    global description_list
    with open('Data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)
pass

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")
pass

def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis=",".join(feature_names).split(",")
        symptoms_present = []

        while True:

        # Prompt the user to enter the symptom
            engine.say("\n Enter the symptom you are experiencing \t\t\t",)
            engine.runAndWait()
            print("\nEnter the symptom you are experiencing  \t\t",end="->")
            disease_input = input("")

            conf,cnf_dis=check_pattern(chk_dis,disease_input)
            if conf==1:
                print("searches related to input: ")
                for num,it in enumerate(cnf_dis):
                    print(num,")",it)
                if num!=0:
                    print(f"Select the one you meant (0 - {num}):  ", end="")
                    conf_inp = int(input(""))
                else:
                    conf_inp=0

                disease_input=cnf_dis[conf_inp]
                break
            else:
                print("Enter valid symptom.")

        while True:
            try:
                user_input = input("Okay. From how many days ? : ")
                if user_input.strip() == "":
                    raise ValueError("Input cannot be empty.")  # Raise an error if input is empty
                num_days = int(user_input)
                break
            except ValueError:
                print("Enter a valid number of days (e.g., 5).")

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])

                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

                engine.say("Are you experiencing any")
                engine.runAndWait()
                print("Are you experiencing any ")
                symptoms_exp=[]
                for syms in list(symptoms_given):
                    inp=""
                    engine.say(f"{syms}, are you experiencing it?")
                    engine.runAndWait()
                    print(syms,"? : ",end='')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if(inp=="yes"):
                        symptoms_exp.append(syms)

                second_prediction=sec_predict(symptoms_exp)
                # print(second_prediction)
                calc_condition(symptoms_exp,num_days)
                if(present_disease[0]==second_prediction[0]):
                    engine.say("You may have ", present_disease[0])
                    engine.runAndWait()
                    print("You may have ", present_disease[0])
                    print(description_list[present_disease[0]])


                else:
                    engine.say(f"You may have {present_disease[0]} or {second_prediction[0]}.")
                    engine.runAndWait()
                    print("You may have ", present_disease[0], "or ", second_prediction[0])
                    print(description_list[present_disease[0]])
                    print(description_list[second_prediction[0]])

                # print(description_list[present_disease[0]])
                precution_list=precautionDictionary[present_disease[0]]
                print("Take following measures : ")
                for  i,j in enumerate(precution_list):
                    print(i+1,")",j)


            recurse(0, 1)
        getSeverityDict()
        getDescription()
        getprecautionDict()
        getInfo()
        tree_to_code(clf,cols)
        print("----------------------------------------------------------------------------------------------------------------------------------")
pass

# Initialize the text-to-speech engine


def text_to_speech(text):
    # Set properties (optional)
    engine.setProperty('rate', 100)    # Speed percent (can go over 100)
    engine.setProperty('volume', 1)  # Volume 0-1

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()
pass

# And so on for other functions you need, e.g. getSeverityDict, getDescription, etc.
