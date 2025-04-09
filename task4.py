# Import necessary libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Task 4: BERT (with fine-tune) for review classification
class AmazonReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def bert_with_finetune(X_train, X_test, y_train, y_test):
    print("\n--- Task 4: BERT with Fine-Tuning ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    ).to(device)
    
    train_texts = X_train.tolist()
    test_texts = X_test.tolist()
    train_labels = y_train.tolist()
    test_labels = y_test.tolist()
    
    train_dataset = AmazonReviewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = AmazonReviewsDataset(test_texts, test_labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=10,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Fine-tuning BERT model...")
    trainer.train()
    
    print("Evaluating fine-tuned BERT model...")
    results = trainer.evaluate()
    
    print(f"Fine-tuned BERT results: {results}")
    
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, preds)
    print(cm)
    
    bert_finetuned_results = {
        'Precision': results['eval_precision'],
        'Recall': results['eval_recall'],
        'Accuracy': results['eval_accuracy'],
        'F1': results['eval_f1']
    }
    
    return bert_finetuned_results