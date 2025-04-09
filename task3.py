# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

def bert_without_finetune(X_test, y_test):
    print("\n--- Task 3: BERT without Fine-Tuning ---")
    classifier = pipeline("sentiment-analysis")
    
    batch_size = 32
    test_texts = X_test.tolist()
    n_batches = (len(test_texts) + batch_size - 1) // batch_size
    
    bert_predictions = []
    
    print("Running BERT prediction on test data...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_texts))
        batch_texts = test_texts[start_idx:end_idx]
        
        truncated_texts = [text[:512] if len(text) > 512 else text for text in batch_texts]
        
        results = classifier(truncated_texts)
        
        batch_predictions = [1 if result['label'] == 'POSITIVE' else 0 for result in results]
        bert_predictions.extend(batch_predictions)
    
    bert_results = {
        'Precision': precision_score(y_test, bert_predictions),
        'Recall': recall_score(y_test, bert_predictions),
        'Accuracy': accuracy_score(y_test, bert_predictions),
        'F1': f1_score(y_test, bert_predictions)
    }
    
    print(classification_report(y_test, bert_predictions))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, bert_predictions)
    print(cm)
    
    return bert_results