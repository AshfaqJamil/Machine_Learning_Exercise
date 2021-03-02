from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('weather.csv')

print(df)

print(df.shape) # prints the instances and the attributes
print(df.head()) #prins the 5 instances of data
print(df.describe()) # prints a little discription of statistical info of data set

X = df.drop(columns=['Play']) # dropping target column from the attributes

y = df['Play']

model = DummyClassifier(strategy='most_frequent',random_state=0) #ZeroR classifier

model.fit(X, y)

predictions = model.predict(X)
score = accuracy_score(y,predictions)

print(score)
print(confusion_matrix(y,predictions))
print(classification_report(y,predictions))