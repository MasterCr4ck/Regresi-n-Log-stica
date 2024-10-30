import pandas as pd

data = pd.read_csv('ruta/a/tu/archivo.csv')
data = data.dropna()  


data = pd.get_dummies(data, drop_first=True)
data.head()
data.info()
data.describe()
data = data.dropna() 
data = pd.get_dummies(data, drop_first=True)
from sklearn.feature_selection import SelectKBest, f_classif
X = data.drop('target', axis=1)  
y = data['target']
selector = SelectKBest(f_classif, k=10)  
X_selected = selector.fit_transform(X, y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importancia de características
importances = model.feature_importances_
indices = selector.get_support(indices=True)
features = X.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
