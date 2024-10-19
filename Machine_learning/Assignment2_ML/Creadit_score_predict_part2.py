from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('credit.csv')

le = LabelEncoder()
df['Debt'] = le.fit_transform(df['Debt'])
df['Income'] = le.fit_transform(df['Income'])
df['Married?'] = le.fit_transform(df['Married?'])
df['Owns_Property'] = le.fit_transform(df['Owns_Property'])
df['Gender'] = le.fit_transform(df['Gender'])

X = df[['Debt','Income','Married?','Owns_Property','Gender']]
Y = df['Risk']

model = DecisionTreeClassifier()
model.fit(X,Y)
tree_rules = export_text(model, feature_names=['debt', 'income', 'marital_status', 'property', 'gender'])
print(tree_rules)

tom = [[0, 0, 0, 1, 1]] # low, low, no, Yes, Male
ana = [[0, 1, 1, 1, 0]]  # low, medium, yes, Yes, female
Sofia = [[1, 0, 0, 0, 0]] # medium, low, no, no, female
fred = [[0, 0, 1, 0, 1]] # low,low,yes,no,male,high

tom_risk = model.predict(tom)
ana_risk = model.predict(ana)
sofia = model.predict(Sofia)
fred = model.predict(fred)

print(f"Tom's has a {tom_risk} risk")
print(f"Ana's has a {ana_risk} risk")
print(f"Fred has a {fred} risk")
print(f"Sofia has a {sofia} risk")