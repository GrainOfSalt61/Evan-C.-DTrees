import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

data = pd.read_csv('cleanfuel.csv')
data.head()
data.info()

#data = data.drop('engine_cylinders', axis=1)
#data = data.drop('fuel_type', axis=1)
data = data.drop('tailpipe_co2_in_grams_mile_ft1', axis=1)

X = data.drop('city_mpg_ft1', axis=1)
y = data['city_mpg_ft1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
test_size=0.2)
#original dTree process
clf = DecisionTreeClassifier(max_depth=8)
print(cross_val_score(clf, X_train, y_train, cv=7))
clf.fit(X_train, y_train)
print(metrics.accuracy_score(y_valid, clf.predict(X_valid)))

#bagging
#clf = DecisionTreeClassifier()
#bagging_model = BaggingClassifier(clf, n_estimators=10)
#bagging_model.fit(X_train, y_train)
#accuracy = bagging_model.score(X_test, y_test)  # Example: Calculate accuracy
#print("Bagging Accuracy: " + str(accuracy))

plt.figure(figsize=(350,60))
_ = tree.plot_tree(clf, feature_names = list(X_train), filled=True, fontsize=20)
plt.savefig('crop.png')

#finding importance
#def sortSecond(val):
#    return val[1]
##values = clf.feature_importances_
#features = list(X)
#importances = [(features[i], values[i]) for i in range(len(features))]
#importances.sort(reverse=True, key=sortSecond)
#print(importances)

#plt.barh(features,values)
#plt.show()