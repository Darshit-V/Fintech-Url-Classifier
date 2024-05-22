import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

with open('mixed_labeled_data.txt', 'r') as file:
    lines = file.readlines()

words = []
labels = []
for line in lines:
    label, word = line.strip().split()
    words.append(word)
    labels.append(int(label))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(words)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = rf_classifier.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

cv_scores = cross_val_score(rf_classifier, X, labels, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

import joblib
joblib.dump(rf_classifier, 'RandomForest_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer_RandomForest.joblib')
