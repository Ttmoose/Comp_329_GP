#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:31:49 2026

@author: raihan
"""

import sklearn
import numpy as np
import pandas as pd
import re
from scipy.sparse import lil_matrix


from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score


data = []
labels = []
artists = []

df = pd.read_csv("/Users/raihan/Desktop/COMP 329/FINAL/songs.csv")


df = df.dropna(subset = ["lyrics", "genre", "artists"])
df = df[df["lyrics"].astype(str).str.len() >= 50]
df= df.drop_duplicates(subset = ["lyrics"]).reset_index(drop = True)

for _, row in df.iterrows():
    text = str(row["lyrics"]).replace("\\n", " ").replace("\n", " ")
    data.append(text.strip())
    labels.append(row["genre"])
    artists.append(row["artists"])

labels = np.array(labels)
artists = np.array(artists)
label_names = sorted(np.unique(labels).tolist())

class CustomVectorizer:
    
    def __init__(self, min_df = 2, max_df = None):
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
    
    def preprocess(self, text):
        tokens = re.findall(r'[a-z]{2,}', text.lower())
        return tokens
    
    def fit(self, documents):
        num_docs = len(documents)
        doc_frequency = {}
        for doc in documents:
            tokens = set(self.preprocess(doc))
            for token in tokens:
                if token in doc_frequency:
                    doc_frequency[token] += 1
                else:
                    doc_frequency[token] = 1
        if self.max_df is not None:
            max_df = self.max_df
        else:
            max_df = num_docs
        filtered = []
        for token, frequency in doc_frequency.items():
            if frequency >= self.min_df and frequency <= max_df:
                filtered.append(token)
        filtered.sort()
        
        self.vocabulary = {}
        for index in range(len(filtered)):
            self.vocabulary[filtered[index]] = index
        return self
    
    def transform(self, documents):
        num_docs = len(documents)
        num_tokens = len(self.vocabulary)
        X = lil_matrix((num_docs, num_tokens), dtype = np.int32)
        for i in range(len(documents)):
            tokens = self.preprocess(documents[i])
            for token in tokens:
                if token in self.vocabulary:
                    X[i, self.vocabulary[token]] += 1
        return X.tocsr()
    
def custom_vectorizer(data, labels, artists, label_names):
        data_arr = np.array(data, dtype = object)
        
        splitter1 = GroupShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 60)
        trainval_idx, test_idx = next(splitter1.split(data_arr, labels, groups = artists))
        
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=60)
        rel_train, rel_dev = next(splitter2.split(data_arr[trainval_idx], labels[trainval_idx], 
                                                  groups=artists[trainval_idx]))
        #learned from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
        
        train_idx = trainval_idx[rel_train]
        dev_idx = trainval_idx[rel_dev]
        
        X_train = data_arr[train_idx]
        y_train = labels[train_idx]
        X_dev = data_arr[dev_idx].tolist()
        y_dev = labels[dev_idx]
        X_test = data_arr[test_idx].tolist()
        y_test = labels[test_idx]
        
        
        custom_vector = CustomVectorizer(min_df = 2)
        custom_vector.fit(X_train)
        X_train_counts = custom_vector.transform(X_train)
        print(X_train_counts.shape)
        X_dev_counts = custom_vector.transform(X_dev)
        X_test_counts = custom_vector.transform(X_test)
        
        best_alpha = 1
        best_dev_accuracy = 0
        for alpha in [.01, .1, 0.5, 1, 1.5, 2]:
            classifier = MultinomialNB(alpha = alpha).fit(X_train_counts, y_train)
            accuracy = np.mean(classifier.predict(X_dev_counts) ==  y_dev)
            if accuracy > best_dev_accuracy:
                best_dev_accuracy = accuracy
                best_alpha = alpha
                
        print(f"Best alpha: {best_alpha}")
        best_model = MultinomialNB(alpha = best_alpha).fit(X_train_counts, y_train)
        
        predicted_test = best_model.predict(X_test_counts)
        test_accuracy = np.mean(predicted_test == y_test)
        test_macro_f1 = f1_score(y_test, predicted_test, average="macro")
        
        print(f"Dev Accuracy: {best_dev_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Macro-F1: {test_macro_f1:.4f}")
        print("\nPer-class report (custom):")
        print(classification_report(y_test, predicted_test, digits=3))

        return X_train, X_dev, X_test, y_train, y_dev, y_test, best_dev_accuracy, test_accuracy, predicted_test
    
def sklearn_vectorizer(X_train, X_dev, X_test, y_train, y_dev, y_test, target_names):
            
        best_sklearn_accuracy = 0
        best_sklearn_pipeline = None
        best_sklearn_alpha = None
            
        for alpha in [.01, .1, 0.5, 1, 1.5, 2]:
            text_clf1 = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha = alpha)), ])
            text_clf1.fit(X_train, y_train) 
            accuracy1 = np.mean(text_clf1.predict(X_dev) == y_dev)
               
            
            if accuracy1 > best_sklearn_accuracy:
                best_sklearn_accuracy = accuracy1
                best_sklearn_pipeline = text_clf1
                best_sklearn_alpha = alpha

            pipe2 = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB(alpha=alpha)), ])
            pipe2.fit(X_train, y_train)
            accuracy2 = np.mean(pipe2.predict(X_dev) == y_dev)
                   
            if accuracy2 > best_sklearn_accuracy:
                best_sklearn_accuracy = accuracy2
                best_sklearn_pipeline = pipe2
                best_sklearn_alpha = alpha
                   
                   
                   
        print(f"Best sklearn alpha: {best_sklearn_alpha}")
        print(f"Best dev accuracy: {best_sklearn_accuracy:.4f}")
        
        predicted_test = best_sklearn_pipeline.predict(X_test)
        test_accuracy = np.mean(predicted_test == y_test)
        test_macro_f1 = f1_score(y_test, predicted_test, average="macro")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test Macro-F1: {test_macro_f1:.4f}")
        print("\nPer-class report (sklearn):")
        print(classification_report(y_test, predicted_test, digits=3))

        return best_sklearn_accuracy, test_accuracy, predicted_test
    
    
X_train, X_dev, X_test, y_train, y_dev, y_test, custom_dev, custom_test, custom_pred = custom_vectorizer(
    data, labels, artists, label_names)
sklearn_dev, sklearn_test, sklearn_pred = sklearn_vectorizer(
    X_train, X_dev, X_test, y_train, y_dev, y_test, label_names)

print(f"Overall Accuracy: {sklearn_test*100:.2f}%")
print("-" * 40)
print("Classification Report:")
print(classification_report(y_test, sklearn_pred, digits=2))