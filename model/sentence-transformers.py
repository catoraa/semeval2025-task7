import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'重试重试

from torch.nn import functional as F
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
strict = False


def verdict_EntailmentOrNot(text1, text2):
    global tokenizer, model, strict
    inputs = tokenizer(text1, text2, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    largest_index = torch.argmax(probabilities, dim=1)
    prediction1 = largest_index
    inputs = tokenizer(text2, text1, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    largest_index = torch.argmax(probabilities, dim=1)
    prediction2 = largest_index
    if strict:
        if prediction1 == prediction2:
            if prediction1 == 2:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        if prediction1 == 2 or prediction2 == 2:
            return 1
        else:
            return 0


text1 = 'it is a nice day'
text2 = 'it is a sunny day'
print(verdict_EntailmentOrNot(text1, text2))
