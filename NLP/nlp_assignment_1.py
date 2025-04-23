import json
import torch
import numpy as np
import re
import string
import collections
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForQuestionAnswering
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import evl
""" 1. Load and preprocess SQuAD dataset"""

def load_squad_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    contexts = []
    questions = []
    answers = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if qa['answers']:
                    answer = qa['answers'][0]
                    contexts.append(context)
                    questions.append(question)
                    answers.append({
                        'text': answer['text'],
                        'answer_start': answer['answer_start']
                    })

    return contexts, questions, answers

"""2. Create custom dataset for SQuAD"""

class SQuADDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_length=384):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]

        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation='only_second',
            stride=128,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )

        answer_text = answer['text']
        start_char = answer['answer_start']
        end_char = start_char + len(answer_text)

        offsets = inputs.pop('offset_mapping')[0].numpy()
        start_token = end_token = 0

        for i, (start, end) in enumerate(offsets):
            if start <= start_char and end > start_char:
                start_token = i
            if start < end_char and end >= end_char:
                end_token = i
                break

        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['start_positions'] = torch.tensor(start_token)
        item['end_positions'] = torch.tensor(end_token)

        return item

"""3. Train the model"""

def train_model(model, train_dataloader, val_dataloader, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_qa_model.pt')

    return model

"""4. Predict answers"""

def predict_answer(model, tokenizer, question, context):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation='only_second',
        stride=128,
        padding='max_length',
        return_tensors='pt',
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop('offset_mapping').cpu().numpy()[0]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits.cpu().numpy()[0]
    end_logits = outputs.end_logits.cpu().numpy()[0]

    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    if end_idx < start_idx:
        end_idx = start_idx

    char_start = offset_mapping[start_idx][0]
    char_end = offset_mapping[end_idx][1]

    answer = context[char_start:char_end]

    return answer

"""5. F1 Score calculation based on the official SQuAD evaluation script"""

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(pred_answer, true_answer):
    pred_tokens = get_tokens(pred_answer)
    true_tokens = get_tokens(true_answer)
    
    if len(true_tokens) == 0 or len(pred_tokens) == 0:
        return 0
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(true_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def calculate_exact_match_and_f1(model, tokenizer, contexts, questions, answers):
    exact_match_score = 0
    f1_score_total = 0
    count = 0
    
    for i, (context, question, answer) in enumerate(zip(contexts, questions, answers)):
        prediction = predict_answer(model, tokenizer, question, context)
        true_answer = answer['text']
        
        exact_match = normalize_answer(prediction) == normalize_answer(true_answer)
        if exact_match:
            exact_match_score += 1
            
        f1 = compute_f1(prediction, true_answer)
        f1_score_total += f1
        
        count += 1
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} examples")
    
    exact_match = 100.0 * exact_match_score / count
    f1 = 100.0 * f1_score_total / count
    
    return exact_match, f1

"""6. Evaluate using the official SQuAD script"""

def evaluate_model(model, tokenizer, eval_data):
    predictions = {}

    for example in tqdm(eval_data):
        question = example['question']
        context = example['context']
        qid = example['id']

        answer = predict_answer(model, tokenizer, question, context)
        predictions[qid] = answer

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

def load_squad_eval_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    eval_examples = []
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                example = {
                    'question': qa['question'],
                    'context': context,
                    'id': qa['id']
                }
                eval_examples.append(example)
    
    return eval_examples

def main():
    print("Loading data...")
    contexts, questions, answers = load_squad_data('train-v2.0.json')

    # Split data
    train_contexts, val_contexts, train_questions, val_questions, train_answers, val_answers = train_test_split(
        contexts, questions, answers, test_size=0.1, random_state=42
    )

    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    print("Creating datasets...")
    train_dataset = SQuADDataset(train_contexts, train_questions, train_answers, tokenizer)
    val_dataset = SQuADDataset(val_contexts, val_questions, val_answers, tokenizer)

    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    print("Training model...")
    trained_model = train_model(model, train_dataloader, val_dataloader)

    print("Model training completed!")
    
    # Load the best model
    model.load_state_dict(torch.load('best_qa_model.pt'))
    
    print("Calculating F1 score on validation set...")
    sample_size = min(500, len(val_contexts))
    sampled_indices = np.random.choice(len(val_contexts), sample_size, replace=False)
    
    sampled_contexts = [val_contexts[i] for i in sampled_indices]
    sampled_questions = [val_questions[i] for i in sampled_indices]
    sampled_answers = [val_answers[i] for i in sampled_indices]
    
    exact_match, f1 = calculate_exact_match_and_f1(model, sampled_contexts, sampled_questions, sampled_answers)
    
    print(f"Exact Match: {exact_match:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print("To evaluate on the official dev set check evaluation.py")
if __name__ == "__main__":
    main()