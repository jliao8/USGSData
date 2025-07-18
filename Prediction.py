import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier

st.title("Earthquake Home")
'''
### K-Nearest Neighbors ###
st.header("K-Nearest Neighbors")
n_neighbors = st.slider("Number of Neighbors:", min_value=3, max_value=100)
KNN_feature = df.drop(["magType","id","time","netlocmagSource"],axis=1,inplace=False)
KNN_target = df["netlocmagSource"]
x_train, x_test, y_train, y_test = train_test_split(KNN_feature, KNN_target, test_size = .3)
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(x_train, y_train)
KNN_pred = model.predict(x_test)
model.score(x_test,y_test)
report = metrics.classification_report(y_test, KNN_pred)
st.text(".\n"+report)

st.divider()

### Decision Tree ###
st.header("Decision Tree")
max_leaf_nodes = st.slider("Max Leaf Nodes:", min_value=2)
max_depth = st.slider("Max Depth:", min_value=1)
dt_feature = df.drop(["magType","id","time","netlocmagSource"],axis=1,inplace=False)
dt_target = df["netlocmagSource"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(dt_feature, dt_target, test_size = .3)
model = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, max_depth=max_depth)
model.fit(x_train, y_train)
model.predict(x_test)
fig = plt.figure(figsize=(16,9))
tree.plot_tree(model)
st.pyplot(fig)
st.write("Score: ", model.score(x_test, y_test))

st.divider()

### K-Means ###
st.header("K-Means")
n_clusters = st.slider("Number of Clusters:", min_value=2, max_value=10)
kmeans_feature = df.drop(["magType","id","time","netlocmagSource"],axis=1,inplace=False)
kmeans_target = df["netlocmagSource"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(kmeans_feature, kmeans_target, test_size = .3)
model = KMeans(n_clusters=n_clusters, n_init='auto')
model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test,y_test)
fig = plt.figure(figsize=(16,9))
sns.scatterplot(data=x_train, x= 'longitude', y= 'latitude',hue=model.labels_)
st.pyplot(fig)'''
