import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import KMeans

# Importing data
koston_file_path = '/Users/jarren/Koston/koston.csv'
koston = pd.read_csv(koston_file_path, sep='\t')

# Data Explortation
# koston.info()
# koston.head()

# small data clean up
koston.drop(columns=['obstacle_detailed', 'location','trick_index', 'clip_index'], inplace=True)
koston.fillna(0, inplace=True)
obstacles_to_keep = ['ledge','stair','gap','rail','flat','manual','transition','hip']
koston = koston[koston.obstacle.isin(obstacles_to_keep) == True]

# Visualizing Data
# plt.hist(koston.obstacle)
# plt.title('obstacle')
# plt.show()
# print('Transition Skating: ', koston['obstacle'].value_counts()['transition'])

# sns.catplot(data=koston, x='switch', y='obstacle', kind='bar', height=3, aspect=3)
# sns.catplot(data=koston, x='line', y='obstacle', kind='bar', height=3, aspect=3)

# Creating Training and Test Set
y = koston.switch
X = koston.drop(columns=['switch'])
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Data Preprocessing
ohe = OneHotEncoder(sparse=False)
koston_1hot = ohe.fit_transform(X_train)

# Pipelining
num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# pipeline for numerical columns
num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

# pipeline for categorical columns
cat_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='N/A'),
    OneHotEncoder(handle_unknown='ignore', sparse=False)
)

full_pipe = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Testing out different algorithms

# Logistic Regression
""""
lr = make_pipeline(
    full_pipe, LogisticRegression(random_state=0))
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

score = accuracy_score(y_val, y_pred)
print("Accuracy Score:", score)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)
"""

# Decision Tree Regressor
"""
dfr = make_pipeline(
    full_pipe, DecisionTreeRegressor())
dfr.fit(X_train, y_train)
y_pred = dfr.predict(X_val)

score = accuracy_score(y_val, y_pred)
print("Accuracy Score:", score)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)
"""

# K Nearest Neighbors
"""
kn = make_pipeline(
    full_pipe,  KNeighborsClassifier(n_neighbors=13))
kn.fit(X_train, y_train)
y_pred = kn.predict(X_val)

score = accuracy_score(y_val, y_pred)
print("Accuracy Score:", score)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)
"""

# Support Vector Machine
s = make_pipeline(
    full_pipe,  svm.SVC(kernel="linear"))
s.fit(X_train, y_train)
y_pred = s.predict(X_val)

score = accuracy_score(y_val, y_pred)
print("Accuracy Score:", score)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)

# K Means Klustering
"""
km = make_pipeline(
    full_pipe,  KMeans(n_clusters=3, init='random', n_init=10))
km.fit(X_train, y_train)
y_pred = km.predict(X_val)

score = accuracy_score(y_val, y_pred)
print("Accuracy Score:", score)
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)
"""