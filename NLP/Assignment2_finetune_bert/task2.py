import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import re

def preprocess_text_for_word2vec(texts):
    tokenized_texts = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = [token for token in text.split() if token]
        tokenized_texts.append(tokens)
    return tokenized_texts

def word2vec_classification(X_train, X_test, y_train, y_test, sampled_data):
    print("\n--- Task 2: Word2Vec Classification ---")
    max_samples = 50000
    if len(sampled_data) > max_samples:
        print(f"Sampling {max_samples} examples for Word2Vec training...")
        sample_for_training = sampled_data.sample(n=max_samples, random_state=42)
        all_texts = sample_for_training['Text'].tolist()
    else:
        all_texts = sampled_data['Text'].tolist()
    
    print("Preprocessing text for Word2Vec...")
    tokenized_texts = preprocess_text_for_word2vec(all_texts)
    
    print("Training Word2Vec model...")
    
    print("Training a new Word2Vec model...")
    try:
        w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1
        )
    except Exception as e:
        print(f"Error training Word2Vec model: {e}")
        print("Trying with smaller vector size...")
        w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=50,
            window=5,
            min_count=5,  
            workers=1,    
            sg=1  
        )
    
    def get_document_vector(text, model):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = [token for token in text.split() if token]
        
        vec = np.zeros(model.vector_size)
        count = 0
        for word in tokens:
            try:
                if word in model.wv:
                    vec += model.wv[word]
                    count += 1
            except Exception:
                pass
        if count != 0:
            vec /= count
        return vec
    
    print("Creating document vectors...")
    X_train_w2v = []
    batch_size = 1000
    for i in range(0, len(X_train), batch_size):
        end = min(i + batch_size, len(X_train))
        batch = X_train[i:end]
        X_train_w2v.extend([get_document_vector(text, w2v_model) for text in batch])
        print(f"Processed {end}/{len(X_train)} training examples...")
    X_train_w2v = np.array(X_train_w2v)
    
    X_test_w2v = []
    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        batch = X_test[i:end]
        X_test_w2v.extend([get_document_vector(text, w2v_model) for text in batch])
        print(f"Processed {end}/{len(X_test)} test examples...")
    X_test_w2v = np.array(X_test_w2v)
    
    results = {}
    
    try:
        print("\nModel 1: Logistic Regression with Word2Vec")
        lr_model_w2v = LogisticRegression(max_iter=1000, random_state=42, n_jobs=1)
        lr_model_w2v.fit(X_train_w2v, y_train)
        
        lr_pred_w2v = lr_model_w2v.predict(X_test_w2v)
        results['word2vec-mode 1'] = {
            'Precision': precision_score(y_test, lr_pred_w2v),
            'Recall': recall_score(y_test, lr_pred_w2v),
            'Accuracy': accuracy_score(y_test, lr_pred_w2v),
            'F1': f1_score(y_test, lr_pred_w2v)
        }
        print(classification_report(y_test, lr_pred_w2v))
    except Exception as e:
        print(f"Error with Logistic Regression model: {e}")
        results['word2vec-mode 1'] = {
            'Precision': 0.0,
            'Recall': 0.0,
            'Accuracy': 0.0,
            'F1': 0.0
        }
    
    try:
        print("\nModel 2: LinearSVC with Word2Vec")
        # Using dual=False for large datasets and adding verbose for monitoring
        svm_model_w2v = LinearSVC(random_state=42, dual=False, max_iter=1000, verbose=1)
        svm_model_w2v.fit(X_train_w2v, y_train)
        
        svm_pred_w2v = svm_model_w2v.predict(X_test_w2v)
        results['word2vec-mode 2'] = {
            'Precision': precision_score(y_test, svm_pred_w2v),
            'Recall': recall_score(y_test, svm_pred_w2v),
            'Accuracy': accuracy_score(y_test, svm_pred_w2v),
            'F1': f1_score(y_test, svm_pred_w2v)
        }
        print(classification_report(y_test, svm_pred_w2v))
    except Exception as e:
        print(f"Error with LinearSVC model: {e}")
        # Add placeholder results for reporting
        results['word2vec-mode 2'] = {
            'Precision': 0.0,
            'Recall': 0.0, 
            'Accuracy': 0.0,
            'F1': 0.0
        }
    
    try:
        print("\nModel 3: Random Forest with Word2Vec")
        rf_model_w2v = RandomForestClassifier(n_estimators=50, max_depth=20, 
                                              random_state=42, n_jobs=1, verbose=1)
        rf_model_w2v.fit(X_train_w2v, y_train)
        
        rf_pred_w2v = rf_model_w2v.predict(X_test_w2v)
        results['word2vec-mode 3'] = {
            'Precision': precision_score(y_test, rf_pred_w2v),
            'Recall': recall_score(y_test, rf_pred_w2v),
            'Accuracy': accuracy_score(y_test, rf_pred_w2v),
            'F1': f1_score(y_test, rf_pred_w2v)
        }
        print(classification_report(y_test, rf_pred_w2v))
    except Exception as e:
        print(f"Error with Random Forest model: {e}")
        results['word2vec-mode 3'] = {
            'Precision': 0.0,
            'Recall': 0.0,
            'Accuracy': 0.0,
            'F1': 0.0
        }
    
    return results