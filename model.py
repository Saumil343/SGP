import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

from google.colab import drive
drive.mount('/content/drive')

Importing data

df = pd.read_csv("./content/drive/MyDrive/RGS/Crop_recommendation.csv")
df



Data PreProcessing


df.isnull().sum()

print(len(df.columns))

df.describe()

df.info()

Exploratory Data Analysis


Checking Null Values

sns.heatmap(df.isnull(),cmap="coolwarm")
plt.show()

Checking If dataset is balanced or not


sns.countplot(y='label',data=df, palette="plasma_r")

Checking Ph values if they fall between 6 to 7

sns.boxplot(y='label',x='ph',data=df)

Preprocessing

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

Checking relation between P and K

sns.heatmap(X.corr())

SCALLING


MIN MAX

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

STANDARD SCALLER

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = StandardScaler()
X_train_scaled_ST = scaler.fit_transform(X_train)

X_test_scaled_ST = scaler.transform(X_test)

KNN MODEL

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

KNN WITH STANDARD SCALER

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled_ST, y_train)
knn.score(X_test_scaled_ST, y_test)
classess = df["label"].unique()
print(classess)
newdata=knn.predict([[20,72,15,36.00415838,56.01334416,7.313517308,134.8596466]])
print(classess[newdata]) 
# print(classification_report(y_test,predicted_values))

DECISION TREE

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(X_train_scaled,y_train)

predicted_values = DecisionTree.predict(X_test_scaled)
# print(predicted_values)
classess = df["label"].unique()
print(classess)
newdata=DecisionTree.predict([[20,72,15,36.00415838,56.01334416,7.313517308,134.8596466]])
print(classess[newdata]) 

# print(newdata)
DecisionTree.score(X_test_scaled, y_test)

DECISION TREE WITH STANDARD SCALER

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="gini",random_state=2)

DecisionTree.fit(X_train_scaled_ST,y_train)

predicted_values = DecisionTree.predict(X_test_scaled_ST)
DecisionTree.score(X_test_scaled_ST, y_test)

y_pred_test = DecisionTree.predict(X_test_scaled_ST)
print("Accuracy : ")
accuracy_score(y_test, y_pred_test)
print(classification_report(y_test,predicted_values))
# x = metrics.accuracy_score(y_test, predicted_values)

DECISION TREE WITH GINI INDEX

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2)

DecisionTree.fit(X_train_scaled_ST,y_train)

predicted_values = DecisionTree.predict(X_test_scaled_ST)
DecisionTree.score(X_test_scaled_ST, y_test)

y_pred_test = DecisionTree.predict(X_test_scaled_ST)
print("Accuracy : ")
classess = df["label"].unique()
print(classess)
newdata=DecisionTree.predict([[90,42,43,20.87,82.00,6.5,	202]])
print(classess[newdata]) 

accuracy_score(y_test, y_pred_test)
print(classification_report(y_test,predicted_values))
# x = metrics.accuracy_score(y_test, predicted_values)

plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), DecisionTree.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
plt.show()

Random Forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train_scaled, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))

from sklearn.metrics import accuracy_score
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = clf.predict(data)
classess = df["label"].unique()
print(classess)
newdata=clf.predict([[60,55,44,23.004459,82.320763,7.840207,	263.964248]])
print(classess[newdata]) 

# print(prediction)
print(classification_report(y_test,predicted_values))

# y_pred_test = clf.predict(X_test_scaled)
# accuracy_score(y_test, y_pred_test)

from yellowbrick.classifier import ClassificationReport
classes=list(targets.values())
visualizer = ClassificationReport(clf, classes=classes, support=True,cmap="Blues")

visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()

SVM

from sklearn.svm import SVC

svc_linear = SVC(kernel = 'linear').fit(X_train_scaled_ST, y_train)
classess = df["label"].unique()
print(classess)
newdata=svc_linear.predict([[20,72,15,36.00415838,56.01334416,7.313517308,134.8596466]])
print(classess[newdata]) 

print("Linear Kernel Accuracy: ",svc_linear.score(X_test_scaled_ST,y_test))

svc_poly = SVC(kernel = 'rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled_ST,y_test))

svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled_ST,y_test))

LOGISTIC REGRESSION


LogReg = LogisticRegression(random_state=2)

LogReg.fit(X_train_scaled_ST,y_train)

predicted_values = LogReg.predict(X_test_scaled_ST)
classess = df["label"].unique()
print(classess)
newdata=LogReg.predict([[60,55,44,23.004459,82.320763,7.840207,	263.964248]])
print("boom = :", classess[newdata]) 

x = metrics.accuracy_score(y_test, predicted_values)
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(y_test,predicted_values))

