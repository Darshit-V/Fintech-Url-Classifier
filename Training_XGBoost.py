import numpy as np
import xgboost as xgb
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

xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)
xgb_classifier.fit(X_train, y_train)

train_accuracy = xgb_classifier.score(X_train, y_train)
test_accuracy = xgb_classifier.score(X_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

cv_scores = cross_val_score(xgb_classifier, X, labels, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

import joblib
joblib.dump(xgb_classifier, 'XGBoost_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer_XGBoost.joblib')
