import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import warnings
from analysis import *
from task1 import *
from task2 import *
from task3 import *
from task4 import *
from task5 import *
warnings.filterwarnings('ignore')

def main():
    try:
        print("Loading the Amazon Fine Food Reviews dataset...")
        data = pd.read_csv('amazon-fine-foods/Reviews.csv')
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    # Step 2: EDA
    data = perform_eda(data)
    
    # Step 3: Create labels
    data = create_labels(data)
    
    # Step 4: Balance dataset if needed
    data = balance_dataset(data)
    
    # Step 5: Sample and split the data
    X_train, X_test, y_train, y_test, sampled_data = sample_and_split_data(data)
    
    # Task 1: TF-IDF Classification
    tfidf_results = tfidf_classification(X_train, X_test, y_train, y_test)
    
    # Task 2: Word2Vec Classification
    word2vec_results = word2vec_classification(X_train, X_test, y_train, y_test, sampled_data)
    
    # Task 3: BERT without Fine-Tuning
    bert_wo_finetune_results = bert_without_finetune(X_test, y_test)
    
    # Task 4: BERT with Fine-Tuning
    bert_finetune_results = bert_with_finetune(X_train, X_test, y_train, y_test)
    
    # Combine all results
    all_results = {**tfidf_results, **word2vec_results, 
                   'BERT w/o fine tune': bert_wo_finetune_results,
                   'BERT fine tune': bert_finetune_results}
    
    # Task 5: Analyze Results
    final_results = analyze_results(all_results)
    
    print("\nAssignment completed successfully! yay....!!!")
    return final_results

if __name__ == "__main__":
    result_table = main()