"""
Interactive Layer 1 Crisis Classifier — Version 1
Type any message to test. Type 'quit' to exit.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/crisis_classifier/best"
BYPASS_THRESHOLD  = 0.75   # → skip LLM entirely, go to Layer 3
MONITOR_THRESHOLD = 0.50   # → elevated monitoring

ID2LABEL = {
    0: "no_crisis",
    1: "suicidal_ideation",
    2: "self_directed_harm",
    3: "sexual_violence",
    4: "domestic_violence",
    5: "acute_crisis",
}

def load_model():
    print("Loading crisis classifier...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print(f"Ready (device: {device})\n")
    return tokenizer, model, device

def classify(text, tokenizer, model, device):
    inputs = tokenizer(
        text, max_length=512, truncation=True,
        padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    crisis_probs = probs[1:]
    best_crisis_idx = crisis_probs.argmax().item() + 1
    crisis_conf = crisis_probs.max().item()
    no_crisis_conf = probs[0].item()

    if crisis_conf >= MONITOR_THRESHOLD:
        category   = ID2LABEL[best_crisis_idx]
        confidence = crisis_conf
        if crisis_conf >= BYPASS_THRESHOLD:
            routing = "BYPASS  → skip LLM, escalate to Layer 3"
        else:
            routing = "MONITOR → allow LLM, elevated watch"
    else:
        category   = "no_crisis"
        confidence = no_crisis_conf
        routing    = "NORMAL  → pass to LLM as usual"

    return category, confidence, routing

def main():
    tokenizer, model, device = load_model()

    print("=" * 60)
    print("  Layer 1 Crisis Classifier — Version 1")
    print("  Type a message. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        category, confidence, routing = classify(text, tokenizer, model, device)

        print(f"\n  Category  : {category}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Routing   : {routing}")
        print("-" * 60)

if __name__ == "__main__":
    main()