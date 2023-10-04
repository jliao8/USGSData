import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# cache the dataset to make it faster
@st.cache_data 
def load_data():
    # the last 6 months taken from https://earthquake.usgs.gov/earthquakes/search/
    file1 = "/home/user/Project/2022-09-01 to 2022-10-01 World Earthquakes.csv"
    file2 = "/home/user/Project/2022-10-01 to 2022-11-16 World Earthquakes.csv"
    file3 = "/home/user/Project/2022-11-16 to 2023-01-05 World Earthquakes.csv"
    file4 = "/home/user/Project/2023-01-05 to 2023-03-01 World Earthquakes.csv"
    df1,df2,df3,df4 = pd.read_csv(file1),pd.read_csv(file2),pd.read_csv(file3),pd.read_csv(file4)
    df = pd.concat([df1[::-1],df2[::-1],df3[::-1],df4[::-1]],ignore_index = True)

    df.drop_duplicates(inplace=True) # dates do overlap although miniscule
    df.drop("type", axis=1, inplace=True) # redundant
    df.drop("updated", axis=1, inplace=True) # no valuable information
    df.drop("place", axis=1, inplace=True) # no need to have a reference point when long and lat are provided
    df.dropna(inplace=True) # drop all rows with a missing value

    for i, r in df.iterrows(): # accurate reviewed data
        if r["status"] == "automatic" or r["status"] == "deleted":
            df.drop(index=i, inplace=True)
    df.drop("status", axis=1, inplace=True) # redundant
    df["time"] = pd.to_datetime(df["time"]) # convert object to datetime

    netlocmag_identical = True
    for i, r in df.iterrows(): # check for redundancy in columns
        if r["locationSource"] != r["magSource"] or r["locationSource"] != r["net"]:
            netlocmag_identical = False
    if netlocmag_identical: # rename the column to combine and drop the others
        df.rename(columns={"net": "netlocmagSource"}, inplace=True) 
        df.drop("magSource", axis=1, inplace=True)
        df.drop("locationSource", axis=1, inplace=True)
    df.reset_index(drop=True, inplace = True)
    return df
df = load_data()

st.title("Earthquake Data Product")
#classification, regression = st.tabs(["Classification","Regression"])
#with st.sidebar:
#    st.write("# Classification")
#    class_algo = st.radio("blah", ('K-Nearest Neighbors','Decision Tree','K-Means','Logistic Regression'), label_visibility='collapsed')
#    st.write("# Regression")
#    reg_algo = st.radio("blah", ['Linear Regression'], label_visibility='collapsed')

st.divider()

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
st.pyplot(fig)
    
st.divider()

### Logistic Regression ###
st.header("Logistic Regression")
n_iterations = st.slider("Number of Iterations:", min_value=0, max_value=500, value=250)
if 'LogKeys' not in st.session_state: # keeps track of which keys our algorithm used
    st.session_state['LogKeys'] = [n_iterations]
elif n_iterations not in st.session_state['LogKeys']:
    st.session_state['LogKeys'].append(n_iterations)

logreg_feature = df.drop(["magType","id","time","netlocmagSource"],axis=1,inplace=False)
logreg_target = df["netlocmagSource"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(logreg_feature, [i[0] for i in logreg_target], test_size=.3)
model = LogisticRegression(solver="lbfgs", max_iter=n_iterations)
model.fit(x_train, y_train)
logreg_predictions = model.predict(x_test)
model_score = model.score(x_test,y_test)

if n_iterations not in st.session_state:
    st.session_state[n_iterations] = model_score
col1, col2 = st.columns([1,1])
with col1:
    table_button = st.button("Results", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)
if table_button:
    for iteration in st.session_state['LogKeys']:
        st.write("Iteration: ", iteration, "Score: ", st.session_state[iteration])
if clear_button:
    for key in st.session_state.keys():
        del st.session_state[key]

st.divider()

### Linear Regression ###
st.header("Linear Regression")
linreg_feature = df.drop(["time","magType","netlocmagSource","id","longitude"], axis=1, inplace=False)
linreg_target = df["latitude"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(linreg_feature, linreg_target, test_size=.3)
model = LinearRegression()
model.fit(x_train, y_train)
linreg_predictions = model.predict(x_test)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
options = st.multiselect('Choose Plots to Display:',['Longitude','Depth','Magnitude','Gap','HorizontalError','Predictions'])
if "Longitude" in options:
    plt.scatter(df["longitude"], linreg_target, c='blue')
if "Depth" in options:
    plt.scatter(df["depth"], linreg_target, c='yellow')
if "Magnitude" in options:
    plt.scatter(df["mag"], linreg_target, c='purple')
if "Gap" in options:
    plt.scatter(df["gap"], linreg_target, c='pink')
if "HorizontalError" in options:
    plt.scatter(df["horizontalError"], linreg_target, c='green')
if "Predictions" in options:
    plt.plot(x_test, linreg_predictions, color='red')
plt.ylabel("Y-Axis")
plt.xlabel("X-Axis")
st.pyplot(fig)
st.divider()
