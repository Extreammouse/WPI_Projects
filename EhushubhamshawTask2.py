#task2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class NewsClassifier:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        X = self.df['text']
        y = self.df['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    def extract_features(self, feature_type='tfidf'):
        """Extract features using different methods"""
        if feature_type == 'count':
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        else:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            
        X_train_features = vectorizer.fit_transform(self.X_train)
        X_test_features = vectorizer.transform(self.X_test)
        
        return X_train_features, X_test_features
    
    def train_evaluate_model(self, model, X_train, X_test):
        """Train model and get evaluation metrics"""
        model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)        
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        return precision, recall, accuracy, conf_matrix
    
    def plot_confusion_matrix(self, conf_matrix, title):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def run_all_models(self):
        """Run all combinations of models and feature sets"""
        models = {
            'MultinomialNB': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'LinearSVC': LinearSVC(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        feature_types = ['count', 'tfidf']
        results = []
        best_models = []
        
        for feature_type in feature_types:
            print(f"\nExtracting {feature_type} features...")
            X_train_features, X_test_features = self.extract_features(feature_type)
            
            for model_name, model in models.items():
                print(f"Training {model_name} with {feature_type} features...")
                precision, recall, accuracy, conf_matrix = self.train_evaluate_model(
                    model, X_train_features, X_test_features
                )
                
                results.append([
                    model_name,
                    feature_type,
                    f"{precision:.3f}",
                    f"{recall:.3f}",
                    f"{accuracy:.3f}"
                ])
                
                best_models.append({
                    'model_name': model_name,
                    'feature_type': feature_type,
                    'accuracy': accuracy,
                    'conf_matrix': conf_matrix
                })
        
        headers = ['Model', 'Features', 'Precision', 'Recall', 'Accuracy']
        print("\nModel Performance Results:")
        print(tabulate(results, headers=headers, tablefmt='grid'))
        best_models.sort(key=lambda x: x['accuracy'], reverse=True)
        for i in range(2):
            model_info = best_models[i]
            title = f"{model_info['model_name']} with {model_info['feature_type']}"
            self.plot_confusion_matrix(model_info['conf_matrix'], title)
            
            print(f"\nDetailed Analysis for {title}")
            print(f"Confusion Matrix:")
            print(model_info['conf_matrix'])
            tn, fp, fn, tp = model_info['conf_matrix'].ravel()
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Positives: {tp}")

if __name__ == "__main__":
    classifier = NewsClassifier('combined_news.csv')    
    print("Preprocessing data...")
    classifier.preprocess_data()
    classifier.run_all_models()