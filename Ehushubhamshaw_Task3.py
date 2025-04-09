#Task3
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
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
            print(f"Please manually download '{resource}' using nltk.download('{resource}')")

download_nltk_resources()

class NewsClassifier:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_data(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        X = self.df['text']
        y = self.df['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    def filter_text_by_pos(self, text, pos_filter):
        """Filter text by POS tags"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            
            if pos_filter == 'nouns':
                filtered_tokens = [word for word, tag in tagged if tag.startswith('NN')]
            elif pos_filter == 'adj_nouns':
                filtered_tokens = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('JJ')]
            elif pos_filter == 'stopword_removed':
                filtered_tokens = [word for word, tag in tagged if word not in self.stop_words]
            else:
                filtered_tokens = tokens
            
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error in POS filtering: {e}")
            return text 
    
    def extract_features(self, feature_type='tfidf', pos_filter=None):
        """Extract features using different methods and filters"""
        print(f"Extracting {feature_type} features with {pos_filter if pos_filter else 'no'} POS filter...")
        
        if pos_filter:
            processed_train = []
            for text in self.X_train:
                processed_train.append(self.filter_text_by_pos(text, pos_filter))
                
            processed_test = []
            for text in self.X_test:
                processed_test.append(self.filter_text_by_pos(text, pos_filter))
        else:
            processed_train = self.X_train
            processed_test = self.X_test
        
        if feature_type == 'count':
            vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        else:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        X_train_features = vectorizer.fit_transform(processed_train)
        X_test_features = vectorizer.transform(processed_test)
        
        return X_train_features, X_test_features
    
    def train_evaluate_model(self, model, X_train, X_test):
        """Train model and get evaluation metrics"""
        model.fit(X_train, self.y_train)
        y_pred = model.predict(X_test)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        return precision, recall, accuracy, conf_matrix
    
    def run_all_models(self):
        """Run all combinations of models and feature sets with POS filters"""
        models = {
            'MultinomialNB': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'LinearSVC': LinearSVC(random_state=42, dual=False),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        feature_types = ['count', 'tfidf']
        pos_filters = [None, 'stopword_removed', 'nouns', 'adj_nouns']
        
        results = []
        
        for model_name, model in models.items():
            for feature_type in feature_types:
                for pos_filter in pos_filters:
                    try:
                        X_train_features, X_test_features = self.extract_features(
                            feature_type=feature_type, 
                            pos_filter=pos_filter
                        )
                    
                        precision, recall, accuracy, _ = self.train_evaluate_model(
                            model, X_train_features, X_test_features
                        )
                    
                        filter_name = pos_filter if pos_filter else "none"
                        results.append([
                            model_name,
                            feature_type,
                            filter_name,
                            f"{precision:.3f}",
                            f"{recall:.3f}",
                            f"{accuracy:.3f}"
                        ])
                    
                        print(f"Completed: {model_name} with {feature_type} features and {filter_name} filter")
                    
                    except Exception as e:
                        print(f"Error with {model_name}, {feature_type}, {pos_filter}: {e}")
        
        results.sort(key=lambda x: float(x[5]), reverse=True)
        
        headers = ['ML Model', 'Feature', 'Filter', 'Precision', 'Recall', 'Accuracy']
        print("\nModel Performance Results:")
        print(tabulate(results, headers=headers, tablefmt='grid'))
        
        results_df = pd.DataFrame(results, columns=headers)
        results_df.to_csv('pos_filtering_results.csv', index=False)
        
        print("\nPerformance Analysis:")
        self.analyze_improvement(results)
        
    def analyze_improvement(self, results):
        """Analyze and report on improvement from baseline to POS filtering"""
        results_df = pd.DataFrame(results, columns=['Model', 'Feature', 'Filter', 'Precision', 'Recall', 'Accuracy'])
        results_df['Accuracy'] = results_df['Accuracy'].astype(float)
        
        for model in results_df['Model'].unique():
            for feature in results_df['Feature'].unique():
                subset = results_df[(results_df['Model'] == model) & (results_df['Feature'] == feature)]
                
                baseline = subset[subset['Filter'] == 'none']['Accuracy'].values
                if len(baseline) == 0:
                    continue
                baseline_acc = baseline[0]
                
                print(f"\n{model} with {feature} features:")
                print(f"  Baseline accuracy: {baseline_acc:.3f}")
                
                for filter_type in ['stopword_removed', 'nouns', 'adj_nouns']:
                    filtered = subset[subset['Filter'] == filter_type]['Accuracy'].values
                    if len(filtered) == 0:
                        continue
                    filtered_acc = filtered[0]
                    
                    diff = filtered_acc - baseline_acc
                    percent = (diff / baseline_acc) * 100
                    
                    if diff > 0:
                        print(f"  {filter_type}: {filtered_acc:.3f} (+{diff:.3f}, +{percent:.2f}%)")
                    else:
                        print(f"  {filter_type}: {filtered_acc:.3f} ({diff:.3f}, {percent:.2f}%)")

if __name__ == "__main__":
    classifier = NewsClassifier('combined_news.csv')
    print("Preprocessing data...")
    classifier.preprocess_data()
    classifier.run_all_models()