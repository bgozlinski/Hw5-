import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def extract_time_domain_features(data):
    features = []
    for axis in ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']:
        features.append(data[axis].mean())
        features.append(data[axis].var())
        features.append(data[axis].std())
        features.append(np.sqrt(np.mean(data[axis]**2)))  # Root mean square
        features.append(np.mean(np.abs(data[axis] - data[axis].mean())))  # Mean absolute deviation
        features.append(data[axis].max())
        features.append(data[axis].min())
        features.append(np.percentile(data[axis], 75) - np.percentile(data[axis], 25))  # Interquartile range
        features.append(data[axis].skew())
        features.append(data[axis].kurt())
    return features


def load_data_from_folder(folder_path, label):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            data = pd.read_csv(filepath)
            features = extract_time_domain_features(data)
            X.append(features)
            y.append(label)
    return X, y


# Ścieżki do folderów
base_path = 'data'
folders = {
    'idle': os.path.join(base_path, 'idle'),
    'running': os.path.join(base_path, 'running'),
    'stairs': os.path.join(base_path, 'stairs'),
    'walking': os.path.join(base_path, 'walking')
}


# Wczytanie danych z folderów
X = []
y = []

for label, folder_path in folders.items():
    X_folder, y_folder = load_data_from_folder(folder_path, label)
    X.extend(X_folder)
    y.extend(y_folder)

X = np.array(X)
y = np.array(y)

# Podział danych na zbiory treningowy i testowy oraz standaryzacja
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Trening modelu Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest")
print(classification_report(y_test, y_pred_rf))
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')

# Trening modelu SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM")
print(classification_report(y_test, y_pred_svm))
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm)}')

# Porównanie wyników
results = pd.DataFrame({
    'Model': ['Random Forest', 'SVM'],
    'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)]
})

print(results)