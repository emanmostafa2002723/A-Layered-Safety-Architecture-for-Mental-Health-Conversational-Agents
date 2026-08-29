# # evaluate_prompt_dataset.py
# # Test crisis classifier on curated prompt dataset (from Excel)
# # Usage: python evaluate_prompt_dataset.py path/to/test_cases.csv

# import csv
# import json
# import sys
# import os
# from typing import List, Dict
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from datetime import datetime

# # ── CONFIG ────────────────────────────────────────────────
# MODEL_PATH = "models/crisis_classifier/best"
# BYPASS_THRESHOLD  = 0.75   # → skip LLM entirely, go to Layer 3
# MONITOR_THRESHOLD = 0.40   # → elevated monitoring (updated from 0.50 based on testing)

# ID2LABEL = {
#     0: "no_crisis",
#     1: "suicidal_ideation",
#     2: "self_directed_harm",
#     3: "sexual_violence",
#     4: "domestic_violence",
#     5: "acute_crisis",
# }

# class CrisisClassifier:
#     def __init__(self, model_path: str = MODEL_PATH):
#         print(f"Loading model from {model_path}...")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         self.model.to(self.device)
#         self.model.eval()
#         print(f"Model loaded on {self.device}\n")

#     def predict(self, text: str) -> Dict:
#         inputs = self.tokenizer(
#             text,
#             max_length=512,
#             truncation=True,
#             padding=True,
#             return_tensors="pt"
#         ).to(self.device)

#         with torch.no_grad():
#             probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]

#         # Get probabilities for all classes
#         all_probs = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))}
        
#         # Crisis probabilities (exclude no_crisis)
#         crisis_probs = probs[1:]
#         best_crisis_idx = crisis_probs.argmax().item() + 1
#         crisis_conf = crisis_probs.max().item()
#         no_crisis_conf = probs[0].item()

#         if crisis_conf >= MONITOR_THRESHOLD:
#             category = ID2LABEL[best_crisis_idx]
#             confidence = crisis_conf
#             if crisis_conf >= BYPASS_THRESHOLD:
#                 routing = "BYPASS"
#             else:
#                 routing = "MONITOR"
#         else:
#             category = "no_crisis"
#             confidence = no_crisis_conf
#             routing = "NORMAL"

#         return {
#             'predicted_category': category,
#             'confidence': round(confidence, 4),
#             'routing': routing,
#             'all_probabilities': all_probs
#         }

# def read_test_cases(file_path: str) -> List[Dict]:
#     """Read test cases from CSV or Excel file"""
#     print(f"Reading test cases from {file_path}...")
    
#     if file_path.endswith('.csv'):
#         # Try multiple encodings
#         encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
#         test_cases = None
#         used_encoding = None
        
#         for encoding in encodings:
#             try:
#                 with open(file_path, 'r', encoding=encoding) as f:
#                     reader = csv.DictReader(f)
#                     test_cases = list(reader)
#                 used_encoding = encoding
#                 break
#             except UnicodeDecodeError:
#                 continue
        
#         if test_cases is None:
#             raise ValueError(f"Could not read file with any encoding: {encodings}")
        
#         print(f"  Encoding used: {used_encoding}")
    
#     elif file_path.endswith(('.xlsx', '.xls')):
#         # Need pandas for Excel files
#         try:
#             import pandas as pd
#             df = pd.read_excel(file_path)
#             test_cases = df.to_dict('records')
#         except ImportError:
#             print("Error: pandas is required for Excel files. Install with: pip install pandas openpyxl")
#             sys.exit(1)
#     else:
#         raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
#     print(f"Found {len(test_cases)} test cases\n")
#     return test_cases

# def evaluate_test_cases(test_cases: List[Dict], classifier: CrisisClassifier):
#     """Run classifier on all test cases and collect results"""
#     results = []
    
#     for i, test in enumerate(test_cases, 1):
#         # Find the prompt text column (handle different possible column names)
#         prompt_text = None
#         for key in test.keys():
#             key_lower = key.lower()
#             if 'prompt' in key_lower or 'text' in key_lower or 'message' in key_lower:
#                 prompt_text = test[key]
#                 break
        
#         if not prompt_text or str(prompt_text).strip() == '':
#             print(f"⚠️  Test case {i}: No prompt text found, skipping...")
#             continue
        
#         # Clean the text (remove any BOM or special characters)
#         prompt_text = str(prompt_text).strip()
        
#         print(f"Testing [{i}/{len(test_cases)}]: {prompt_text[:80]}...")
        
#         # Get prediction
#         prediction = classifier.predict(prompt_text)
        
#         # Get expected category (handle different possible column names)
#         expected_category = None
#         for key in test.keys():
#             if 'categor' in key.lower():
#                 expected_category = str(test[key]).strip()
#                 break
        
#         # Create result row
#         result = {
#             **test,  # Keep original test case data
#             'Predicted_Category': prediction['predicted_category'],
#             'Confidence': prediction['confidence'],
#             'Routing': prediction['routing'],
#             'Prob_no_crisis': prediction['all_probabilities']['no_crisis'],
#             'Prob_suicidal_ideation': prediction['all_probabilities']['suicidal_ideation'],
#             'Prob_self_directed_harm': prediction['all_probabilities']['self_directed_harm'],
#             'Prob_sexual_violence': prediction['all_probabilities']['sexual_violence'],
#             'Prob_domestic_violence': prediction['all_probabilities']['domestic_violence'],
#             'Prob_acute_crisis': prediction['all_probabilities']['acute_crisis'],
#             'Match': prediction['predicted_category'] == expected_category
#         }
        
#         results.append(result)
        
#         # Print result summary
#         match_symbol = "✓" if result['Match'] else "✗"
#         print(f"  {match_symbol} Expected: {expected_category or 'N/A':20s} → Predicted: {prediction['predicted_category']:20s} ({prediction['confidence']:.1%})\n")
    
#     return results

# def save_results(results: List[Dict], output_path: str):
#     """Save results to CSV and JSON"""
#     # Save CSV
#     csv_path = output_path.replace('.json', '.csv') if output_path.endswith('.json') else output_path + '.csv'
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         if results:
#             fieldnames = list(results[0].keys())
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(results)
#     print(f"\nResults saved to: {csv_path}")
    
#     # Save JSON (for easy analysis)
#     json_path = csv_path.replace('.csv', '.json')
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     print(f"Results saved to: {json_path}")

# def print_summary(results: List[Dict]):
#     """Print summary statistics"""
#     if not results:
#         return
    
#     total = len(results)
#     correct = sum(1 for r in results if r.get('Match', False))
    
#     print("\n" + "="*60)
#     print("EVALUATION SUMMARY")
#     print("="*60)
#     print(f"Total test cases: {total}")
#     print(f"Correct predictions: {correct}")
#     print(f"Overall accuracy: {correct/total:.1%}")
    
#     # Per-category accuracy
#     categories = {}
#     for r in results:
#         # Find expected category
#         expected_cat = None
#         for key in r.keys():
#             if 'categor' in key.lower():
#                 expected_cat = str(r[key]).strip()
#                 break
        
#         cat = expected_cat or 'Unknown'
#         if cat not in categories:
#             categories[cat] = {'total': 0, 'correct': 0}
#         categories[cat]['total'] += 1
#         if r.get('Match', False):
#             categories[cat]['correct'] += 1
    
#     print("\nPer-category accuracy:")
#     for cat, stats in sorted(categories.items()):
#         acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
#         print(f"  {cat:25s}: {stats['correct']}/{stats['total']} ({acc:.1%})")
    
#     # Routing distribution
#     routing_dist = {}
#     for r in results:
#         route = r.get('Routing', 'Unknown')
#         routing_dist[route] = routing_dist.get(route, 0) + 1
    
#     print("\nRouting distribution:")
#     for route, count in sorted(routing_dist.items()):
#         print(f"  {route:15s}: {count}")
    
#     # Confidence distribution
#     high_conf = sum(1 for r in results if r.get('Confidence', 0) >= 0.75)
#     med_conf = sum(1 for r in results if 0.40 <= r.get('Confidence', 0) < 0.75)
#     low_conf = sum(1 for r in results if r.get('Confidence', 0) < 0.40)
    
#     print("\nConfidence distribution:")
#     print(f"  High (≥75%): {high_conf}")
#     print(f"  Medium (40-74%): {med_conf}")
#     print(f"  Low (<40%): {low_conf}")

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python evaluate_prompt_dataset.py <path_to_csv_or_excel>")
#         print("Example: python evaluate_prompt_dataset.py test_prompts.csv")
#         sys.exit(1)
    
#     file_path = sys.argv[1]
    
#     if not os.path.exists(file_path):
#         print(f"Error: File not found: {file_path}")
#         sys.exit(1)
    
#     # Load classifier
#     classifier = CrisisClassifier()
    
#     # Read test cases
#     test_cases = read_test_cases(file_path)
    
#     # Evaluate
#     results = evaluate_test_cases(test_cases, classifier)
    
#     # Save results with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = f"evaluation_results_{timestamp}.json"
#     save_results(results, output_path)
    
#     # Print summary
#     print_summary(results)

# if __name__ == "__main__":
#     main()

# fix_and_evaluate.py
# Fix CSV label mismatches and re-evaluate crisis classifier
# Usage: python fix_and_evaluate.py test_prompts.csv

import csv
import json
import sys
import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

# ── LABEL MAPPING ─────────────────────────────────────────
# Map test CSV labels to model's actual labels
LABEL_FIXES = {
    'self_harm': 'self_directed_harm',
    'anxiety_crisis': 'acute_crisis',
    'substance_abuse': 'acute_crisis',
    'suicide': 'suicidal_ideation',
    'suicidal': 'suicidal_ideation',
    'self-directed_harm': 'self_directed_harm',
    'acute_anxiety': 'acute_crisis',
}

# Categories to skip (not model's job)
SKIP_CATEGORIES = {
    'violence',  # Dropped from training
    'prompt_injection',  # Wrong component - not crisis classifier
    'injection',  # Same as above
    'risk_taking',  # Not in V1 model
}

# Valid model categories
VALID_CATEGORIES = {
    'no_crisis',
    'suicidal_ideation',
    'self_directed_harm',
    'sexual_violence',
    'domestic_violence',
    'acute_crisis',
}

MODEL_PATH = "models/crisis_classifier/best"
BYPASS_THRESHOLD  = 0.75
MONITOR_THRESHOLD = 0.40

ID2LABEL = {
    0: "no_crisis",
    1: "suicidal_ideation",
    2: "self_directed_harm",
    3: "sexual_violence",
    4: "domestic_violence",
    5: "acute_crisis",
}

class CrisisClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        print(f"Loading model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}\n")

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]

        all_probs = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))}
        
        crisis_probs = probs[1:]
        best_crisis_idx = crisis_probs.argmax().item() + 1
        crisis_conf = crisis_probs.max().item()
        no_crisis_conf = probs[0].item()

        if crisis_conf >= MONITOR_THRESHOLD:
            category = ID2LABEL[best_crisis_idx]
            confidence = crisis_conf
            routing = "BYPASS" if crisis_conf >= BYPASS_THRESHOLD else "MONITOR"
        else:
            category = "no_crisis"
            confidence = no_crisis_conf
            routing = "NORMAL"

        return {
            'predicted_category': category,
            'confidence': round(confidence, 4),
            'routing': routing,
            'all_probabilities': all_probs
        }

def normalize_category(category: str) -> str:
    """Normalize category names to match model's labels"""
    if not category:
        return category
    
    # Clean the category
    category = str(category).strip().lower()
    
    # Check if should be skipped
    if category in SKIP_CATEGORIES:
        return None
    
    # Apply direct fixes
    if category in LABEL_FIXES:
        return LABEL_FIXES[category]
    
    # Try fuzzy matching
    if category in VALID_CATEGORIES:
        return category
    
    # Unknown category
    return None

def read_test_cases(file_path: str) -> List[Dict]:
    """Read test cases from CSV with encoding fallback"""
    print(f"Reading test cases from {file_path}...")
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    test_cases = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                test_cases = list(reader)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    
    if test_cases is None:
        raise ValueError(f"Could not read file with any encoding: {encodings}")
    
    print(f"  Encoding: {used_encoding}")
    print(f"  Raw cases: {len(test_cases)}")
    
    # Filter and fix cases
    fixed_cases = []
    skipped_cases = []
    
    for i, case in enumerate(test_cases, 1):
        # Find category column
        category_col = None
        for key in case.keys():
            if 'categor' in key.lower():
                category_col = key
                break
        
        if not category_col:
            print(f"  ⚠️  Row {i}: No category column found, skipping")
            continue
        
        original_category = str(case[category_col]).strip()
        fixed_category = normalize_category(original_category)
        
        if fixed_category is None:
            skipped_cases.append((i, original_category))
            continue
        
        # Update the category
        case[category_col] = fixed_category
        case['Original_Category'] = original_category  # Keep original for reference
        
        # Find prompt text
        prompt_text = None
        for key in case.keys():
            if 'prompt' in key.lower() or 'text' in key.lower() or 'message' in key.lower():
                prompt_text = case[key]
                break
        
        if prompt_text and str(prompt_text).strip():
            fixed_cases.append(case)
    
    print(f"  Valid cases: {len(fixed_cases)}")
    if skipped_cases:
        print(f"  Skipped {len(skipped_cases)} cases:")
        for idx, cat in skipped_cases:
            print(f"    - Row {idx}: '{cat}' (not in V1 model)")
    
    return fixed_cases

def evaluate_test_cases(test_cases: List[Dict], classifier: CrisisClassifier):
    """Run classifier on all test cases"""
    results = []
    
    for i, test in enumerate(test_cases, 1):
        # Find prompt text
        prompt_text = None
        for key in test.keys():
            if 'prompt' in key.lower() or 'text' in key.lower() or 'message' in key.lower():
                prompt_text = str(test[key]).strip()
                break
        
        if not prompt_text:
            continue
        
        # Find expected category
        expected_category = None
        for key in test.keys():
            if 'categor' in key.lower():
                expected_category = str(test[key]).strip()
                break
        
        print(f"[{i}/{len(test_cases)}] Testing: {prompt_text[:70]}...")
        
        prediction = classifier.predict(prompt_text)
        
        result = {
            **test,
            'Predicted_Category': prediction['predicted_category'],
            'Confidence': prediction['confidence'],
            'Routing': prediction['routing'],
            'Prob_no_crisis': prediction['all_probabilities']['no_crisis'],
            'Prob_suicidal_ideation': prediction['all_probabilities']['suicidal_ideation'],
            'Prob_self_directed_harm': prediction['all_probabilities']['self_directed_harm'],
            'Prob_sexual_violence': prediction['all_probabilities']['sexual_violence'],
            'Prob_domestic_violence': prediction['all_probabilities']['domestic_violence'],
            'Prob_acute_crisis': prediction['all_probabilities']['acute_crisis'],
            'Match': prediction['predicted_category'] == expected_category
        }
        
        results.append(result)
        
        match_symbol = "✓" if result['Match'] else "✗"
        print(f"  {match_symbol} Expected: {expected_category:20s} → Got: {prediction['predicted_category']:20s} ({prediction['confidence']:.1%})\n")
    
    return results

def save_results(results: List[Dict], original_filename: str):
    """Save results to CSV and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    
    # Save CSV
    csv_path = f"evaluation_{base_name}_fixed_{timestamp}.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")
    
    # Save JSON
    json_path = csv_path.replace('.csv', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path}")
    
    return csv_path

def print_summary(results: List[Dict]):
    """Print detailed summary"""
    if not results:
        print("No results to summarize")
        return
    
    total = len(results)
    correct = sum(1 for r in results if r.get('Match', False))
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY (Fixed Labels)")
    print("="*70)
    print(f"Total test cases: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {correct/total:.1%}")
    
    # Per-category accuracy
    categories = {}
    for r in results:
        cat = None
        for key in r.keys():
            if 'categor' in key.lower():
                cat = str(r[key]).strip()
                break
        
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if r.get('Match', False):
            categories[cat]['correct'] += 1
    
    print("\nPer-category accuracy:")
    for cat, stats in sorted(categories.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cat:25s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1%})")
    
    # Routing distribution
    routing_dist = {}
    for r in results:
        route = r.get('Routing', 'Unknown')
        routing_dist[route] = routing_dist.get(route, 0) + 1
    
    print("\nRouting distribution:")
    for route, count in sorted(routing_dist.items()):
        print(f"  {route:15s}: {count}")
    
    # Show false negatives by category (important for safety)
    print("\n⚠️  FALSE NEGATIVES (crisis → predicted no_crisis):")
    false_negatives = [r for r in results if not r.get('Match') and r.get('Predicted_Category') == 'no_crisis']
    for r in false_negatives[:10]:  # Show first 10
        prompt = None
        for key in r.keys():
            if 'prompt' in key.lower() or 'text' in key.lower():
                prompt = str(r[key])[:60]
                break
        expected = None
        for key in r.keys():
            if 'categor' in key.lower():
                expected = str(r[key])
                break
        print(f"  Expected: {expected:20s} | {prompt}...")
    
    if len(false_negatives) > 10:
        print(f"  ... and {len(false_negatives) - 10} more")
    
    # Show confusion between crisis categories
    print("\nCategory confusion (predicted wrong crisis type):")
    confusions = [r for r in results if not r.get('Match') and r.get('Predicted_Category') != 'no_crisis']
    for r in confusions[:10]:
        prompt = None
        for key in r.keys():
            if 'prompt' in key.lower() or 'text' in key.lower():
                prompt = str(r[key])[:60]
                break
        expected = None
        for key in r.keys():
            if 'categor' in key.lower():
                expected = str(r[key])
                break
        predicted = r.get('Predicted_Category')
        print(f"  {expected:20s} → {predicted:20s} | {prompt}...")
    
    if len(confusions) > 10:
        print(f"  ... and {len(confusions) - 10} more")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_and_evaluate.py <path_to_csv>")
        print("Example: python fix_and_evaluate.py test_prompts.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Load classifier
    classifier = CrisisClassifier()
    
    # Read and fix test cases
    test_cases = read_test_cases(file_path)
    
    if not test_cases:
        print("No valid test cases found after filtering")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_test_cases(test_cases, classifier)
    
    # Save results
    save_results(results, file_path)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()