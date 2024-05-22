import numpy as np
from sklearn.svm import SVC
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

svm_classifier = SVC(kernel='linear', random_state=42)  
svm_classifier.fit(X_train, y_train)

train_accuracy = svm_classifier.score(X_train, y_train)
test_accuracy = svm_classifier.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

cv_scores = cross_val_score(svm_classifier, X, labels, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

import joblib
joblib.dump(svm_classifier, 'SVM_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer_SVM.joblib')
