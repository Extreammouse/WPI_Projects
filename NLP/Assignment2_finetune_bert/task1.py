# Import necessary libraries
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from analysis import *
import warnings
warnings.filterwarnings('ignore')

def tfidf_classification(X_train, X_test, y_train, y_test):
    print("\n--- Task 1: TF-IDF Classification ---")
    
    results = {}
    
    print("\nModel 1: Logistic Regression with TF-IDF")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    
    lr_pred = lr_model.predict(X_test_tfidf)
    results['TFIDF-model 1'] = {
        'Precision': precision_score(y_test, lr_pred),
        'Recall': recall_score(y_test, lr_pred),
        'Accuracy': accuracy_score(y_test, lr_pred),
        'F1': f1_score(y_test, lr_pred)
    }
    print(classification_report(y_test, lr_pred))
    
    print("\nModel 2: LinearSVC with TF-IDF")
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    
    svm_pred = svm_model.predict(X_test_tfidf)
    results['TFIDF-model 2'] = {
        'Precision': precision_score(y_test, svm_pred),
        'Recall': recall_score(y_test, svm_pred),
        'Accuracy': accuracy_score(y_test, svm_pred),
        'F1': f1_score(y_test, svm_pred)
    }
    print(classification_report(y_test, svm_pred))
    
    print("\nModel 3: Random Forest with TF-IDF")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    
    rf_pred = rf_model.predict(X_test_tfidf)
    results['TFIDF-model 3'] = {
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'Accuracy': accuracy_score(y_test, rf_pred),
        'F1': f1_score(y_test, rf_pred)
    }
    print(classification_report(y_test, rf_pred))
    
    return results