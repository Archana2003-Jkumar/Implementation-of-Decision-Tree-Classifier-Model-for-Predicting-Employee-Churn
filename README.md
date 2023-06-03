# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and read the csv file.
2. Analyse the data .
3. Use label encoder on salary data.
4. Use decision tree for predicting the values and accuracy.
5. Display the results.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: J.Archana priya 
RegisterNumber: 212221230007
*/
```
```
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain ,ytest = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(xtrain,ytrain)
ypred = dt.predict(xtest)

from sklearn import metrics
accuracy = metrics.accuracy_score(ytest,ypred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
### data.head()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/6a8ab4cf-a2ae-4bc9-ab02-396ab839fd5e)
### data.info()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/6b82d49e-d78d-4d89-ad6f-9eafa1e3e4cb)
### isnull() and sum()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/964fb59d-2c41-4480-b14c-4c2efd18b6aa)
### data value counts()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/1f3db0fe-fc58-4129-929c-0b9429b8102c)
### data.head() for salary
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/ad503d23-66d0-41ee-a1fa-3c0add0a015f)
### x.head()
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/16c3ac69-8438-4f90-9a7f-6130c120c2bf)
### accuracy value
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/cbc85e7d-4bf2-40f4-9178-01cc9d3dbff3)
### data prediction
![image](https://github.com/Archana2003-Jkumar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427594/16c75f0d-e1db-4b1e-afce-4ae72652c967)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
