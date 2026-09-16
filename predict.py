import torch
from transformers import AutoTokenizer
from review_classifier import ReviewClassifier

def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs)
        pred = torch.argmax(logits, dim=1).item()
    return "positive" if pred == 1 else "negative"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('model/')
    model = ReviewClassifier(model_path='model/').to(device)
    model.classifier.load_state_dict(torch.load('model/classifier.pth', map_location=device))
    print(predict("This movie was amazing!", model, tokenizer, device))