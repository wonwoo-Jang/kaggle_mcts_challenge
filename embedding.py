from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

import pandas as pd
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def chunk_and_embed(text, chunk_size=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    all_embeddings = []
    total_tokens = tokens['input_ids'].shape[1]
    chunk_start = 0
    while chunk_start < total_tokens:
        chunk_end = min(chunk_start + chunk_size, total_tokens)
        inputs = {k: v[:, chunk_start:chunk_end] for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embedding)
        chunk_start += chunk_size

    final_embedding = torch.mean(torch.stack(all_embeddings), dim=0)

    return final_embedding

train_data = pd.read_csv('train.csv')
lud_rules = train_data['LudRules']

# print(lud_rules.str.replace(' ', '').str.len().max())
print(chunk_and_embed(lud_rules[0]))