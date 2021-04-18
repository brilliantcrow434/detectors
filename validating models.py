#cross validating models
#Create a pipeline that preprocesses the data, trains the model, and then evaluates it
#using cross-validation:

#importing libraries
""""
from sklearn import  datasets
from sklearn import  metrics
from sklearn.model_selection import  KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#load digit dataset
digits = datasets.load_digits()

#create feature matrix
features = digits.data

#create target vector
target = digits.target

#create Standardizer
standardizer = StandardScaler()

# Create logistic regression object

logistic_model = LogisticRegression()

# Create a pipeline that standardizes, then runs logistic regression

pipeline = make_pipeline(standardizer,logistic_model)

# Create k-Fold cross-validation

kf = KFold(n_splits=10,shuffle=True,random_state=1)

# Conduct k-fold cross-validation
cv_results = cross_val_score(pipeline,
                             features,
                             target,
                             cv=kf,
                             scoring="accuracy",
                             n_jobs=1)
#find the mean result
print(cv_results.mean())
print(cv_results)


#creating a baseline regression model to compare against your model
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

#loading data
boston = load_boston()

#create fetures
features, target = boston.data, boston.target

# Make test and training split
features_train, features_test, target_train, target_test = train_test_split(
 features, target, random_state=0)

#create dummy regressor
dummy = DummyRegressor(strategy= "mean")

#train dummy regressor
dummy.fit(features_train,target_train)

#get R-square score
print(dummy.score(features_test,target_test))

#compare model

from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(features_train,target_train)

#get R-scored
print(ols.score(features_test,target_test))
"""



#to evaluate a model using cross entrophy

# Load libraries

"""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
# Generate features matrix and target vector
X, y = make_classification(n_samples = 10000,
 n_features = 3,
 n_informative = 3,
 n_redundant = 0,
 n_classes = 2,
 random_state = 1)
# Create logistic regression
logit = LogisticRegression()
# Cross-validate model using accuracy
print(cross_val_score(logit, X, y, scoring="accuracy"))

# Cross-validate model using precision
print(cross_val_score(logit, X, y, scoring="precision"))

# Cross-validate model using recall
print(cross_val_score(logit, X, y, scoring="recall"))

# Cross-validate model using f1 this is creating a balance between recall and precision
print(cross_val_score(logit,X,y,scoring = "f1"))
"""

# Load libraries
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
# Create feature matrix and target vector
features, target = make_classification(n_samples=10000,
 n_features=10,
n_classes=2,
n_informative=3,
random_state=3)
# Split into training and test sets
features_train, features_test, target_train, target_test = train_test_split(
 features, target, test_size=0.1, random_state=1)

# Create classifier
logit = LogisticRegression()
# Train model
logit.fit(features_train, target_train)
# Get predicted probabilities
target_probabilities = logit.predict_proba(features_test)[:,1]
# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
 target_probabilities)
# Plot ROC curve
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
#plt.show()

# Get predicted probabilities
logit.predict_proba(features_test)[0:1]

print("Threshold:", threshold[116])
print("True Positive Rate:", true_positive_rate[116])
print("False Positive Rate:", false_positive_rate[116])

"""
"""
11.7 Visualizing a Classifier’s Performance
Problem
Given predicted classes and true classes of the test data, you want to visually compare
the model’s quality.
Solution
Use a confusion matrix, which compares predicted classes and true classes:
"""

# Load libraries
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
# Load data
iris = datasets.load_iris()
# Create feature matrix
features = iris.data
# Create target vector
target = iris.target
# Create list of target class names
class_names = iris.target_names
# Create training and test set
features_train, features_test, target_train, target_test = train_test_split(
 features, target, random_state=1)

# Create logistic regression
classifier = LogisticRegression()

# Train model and make predictions
target_predicted = classifier.fit(features_train,
 target_train).predict(features_test)
# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()
"""


#to evaluate regression model using mean_square_error

# Load libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
 n_features = 3,
 n_informative = 3,
 n_targets = 1,
 noise = 50,
 coef = False,
 random_state = 1)

# Create a linear regression object
ols = LinearRegression()

#evaluate model

print(cross_val_score(ols,features,target,scoring = "neg_mean_square_error"))

#NB use mse for linear model and silhouette_score for evaluating how good your model is

