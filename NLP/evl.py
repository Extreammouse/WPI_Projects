import json
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForQuestionAnswering
from nlp_assignment_1 import load_squad_data, calculate_exact_match_and_f1, load_squad_eval_data, evaluate_model

def main():
    contexts, questions, answers = load_squad_data('train-v2.0.json')
    split = int(len(contexts) * 0.9)
    val_contexts   = contexts[split:]
    val_questions  = questions[split:]
    val_answers    = answers[split:]

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model     = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load('best_qa_model.pt', map_location='cpu'))

    exact_match, f1 = calculate_exact_match_and_f1(
        model,
        tokenizer,
        val_contexts,
        val_questions,
        val_answers
    )
    print(f"Validation Exact Match: {exact_match:.2f}%")
    print(f"Validation   F1 Score: {f1:.2f}%")

    dev_data = load_squad_eval_data('dev-v2.0.json')
    evaluate_model(model, tokenizer, dev_data)
    print("Wrote predictions.json for the official SQuAD evaluation script.")

if __name__ == "__main__":
    main()
