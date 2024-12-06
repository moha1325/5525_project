import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# train and test on our feature set generated using librosa
# data = pd.read_csv('datasets/gtzan_processed/processed_features.csv')
# X = data.drop(columns=['genre', 'file_name'])
# y = data['genre']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train and test on 30 second feature set that came with gtzan
# data = pd.read_csv('datasets/gtzan/features_30_sec.csv')
# X = data.drop(columns=['label', 'filename'])
# y = data['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train on the gtzan 3 second feature set, test on the gtzan 30 second feature set
data3 = pd.read_csv('datasets/gtzan/features_3_sec.csv')
X3 = data3.drop(columns=['label', 'filename'])
y3 = data3['label']
id = np.floor(np.arange(9990)/10)
gss = GroupShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)
for train_idx3, test_idx3 in gss.split(data3, groups=id):
    X_train = X3.iloc[train_idx3]
    y_train = y3.iloc[train_idx3]
test_idx30 = test_idx3[test_idx3 % 10 == 0]/10
data30 = pd.read_csv('datasets/gtzan/features_30_sec.csv')
X30 = data30.drop(columns=['label', 'filename'])
y30 = data30['label']
for i in test_idx30:
    X_test = X30.iloc[test_idx30]
    y_test = y30.iloc[test_idx30]

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# all have the same flow 
# initialize model -> train -> get accuracy

# k nearest neighbors
def knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))

# svm
def svm(X_train, y_train, X_test, y_test):
    # rbf kernel
    svm = SVC(kernel='rbf', C=1, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))

# random forest
def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    print("knn results")
    knn(X_train, y_train, X_test, y_test)

    print("\nsvm results")
    svm(X_train, y_train, X_test, y_test)

    print("\nrandom forest results")
    random_forest(X_train, y_train, X_test, y_test)
