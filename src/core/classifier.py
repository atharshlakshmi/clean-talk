from pathlib import Path
project_root = Path(__file__).parent.parent.parent

import torch
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer


id2label = {0: 'safe',
 1: 'adversarial_harmful',
 2: 'vanilla_harmful',
 3: 'adversarial_benign',
 4: 'unsafe',
 5: 'vanilla_benign'}

label2id = {'safe': 0,
 'adversarial_harmful': 1,
 'vanilla_harmful': 2,
 'adversarial_benign': 3,
 'unsafe': 4,
 'vanilla_benign': 5}


final_model_path = project_root / 'models' / 'best_model.pt'

model_path = 'google-bert/bert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                           num_labels=len(id2label),
                                                           id2label=id2label,
                                                           label2id=label2id)
model.load_state_dict(torch.load(str(final_model_path), map_location=device))
model.to(device)
model.eval()

def tokenize_for_inference(prompt_text):
    inputs = tokenizer(prompt_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return inputs.to(device)

def classify_prompt(prompt_text):
    with torch.no_grad():
        inputs = tokenize_for_inference(prompt_text)
        outputs = model(**inputs) 
        logits = outputs.logits[0]

        preds = torch.argmax(logits, dim=0)
        predicted_label_id = preds.item()
  
        probabilities = torch.softmax(logits, dim=0)
        confidence_score = probabilities[predicted_label_id].item()
        return id2label[predicted_label_id], confidence_score