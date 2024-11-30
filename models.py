import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load features
data = pd.read_csv('datasets/gtzan_processed/processed_features.csv')

# features and targets
X = data.drop(columns=['genre', 'file_name'])
y = data['genre']

# uncomment out next 3 lines to test with full feature set that came with gtzan
# data = pd.read_csv('datasets/gtzan/features_30_sec.csv')
# X = data.drop(columns=['label', 'filename'])
# y = data['label']

# splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    print("knn accuracy:", accuracy_score(y_test, y_pred))

# svm
def svm(X_train, y_train, X_test, y_test):
    # rbf kernel
    svm = SVC(kernel='rbf', C=1, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("sVM accuracy:", accuracy_score(y_test, y_pred))

# random forest
def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("random forest accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    print("knn results")
    knn(X_train, y_train, X_test, y_test)

    print("\nsvm results")
    svm(X_train, y_train, X_test, y_test)

    print("\nrandom forest results")
    random_forest(X_train, y_train, X_test, y_test)
