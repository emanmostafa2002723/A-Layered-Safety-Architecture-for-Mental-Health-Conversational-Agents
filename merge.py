import json
import random
from datasets import load_dataset
from collections import Counter

# ── LABEL MAPS ───────────────────────────────────────────

ARNAIZ_MAP = {
    'suicidal_ideation':             'suicidal_ideation',
    'self_harm':                     'self_directed_harm',
    'risk_taking_behaviours':        'self_directed_harm',
    'risk_taking':                   'self_directed_harm',
    'violent_thoughts':              'other_violence',
    'violence':                      'other_violence',
    'anxiety_crisis':                'acute_crisis',
    'substance_abuse_or_withdrawal': 'acute_crisis',
    'substance_abuse':               'acute_crisis',
    'no_crisis':                     'no_crisis',
}

CRADLE_MAP = {
    'suicideideation_active_ongoing':  'suicidal_ideation',
    'suicideideation_passive_ongoing': 'suicidal_ideation',
    'suicideideation_active_past':     'suicidal_ideation',
    'suicideideation_passive_past':    'suicidal_ideation',
    'selfharm_ongoing':                'self_directed_harm',
    'selfharm_past':                   'self_directed_harm',
    'domesticviolence_ongoing':        'domestic_violence',
    'domesticviolence_past':           'domestic_violence',
    'rape_ongoing':                    'sexual_violence',
    'rape_past':                       'sexual_violence',
    'sexualharassment_ongoing':        'sexual_violence',
    'sexualharassment_past':           'sexual_violence',
    'childabuse_endangerment_ongoing': 'other_violence',
    'childabuse_endangerment_past':    'other_violence',
    'no_crisis':                       'no_crisis',
}

# For multi-label CRADLE examples — most severe label wins
CRADLE_PRIORITY = [
    'suicideideation_active_ongoing',
    'suicideideation_active_past',
    'suicideideation_passive_ongoing',
    'suicideideation_passive_past',
    'selfharm_ongoing',
    'selfharm_past',
    'rape_ongoing',
    'rape_past',
    'sexualharassment_ongoing',
    'sexualharassment_past',
    'domesticviolence_ongoing',
    'domesticviolence_past',
    'childabuse_endangerment_ongoing',
    'childabuse_endangerment_past',
    'no_crisis',
]

def pick_cradle_label(raw_labels):
    parts = [p.strip() for p in raw_labels.split(',')] if isinstance(raw_labels, str) else raw_labels
    for priority_label in CRADLE_PRIORITY:
        if priority_label in parts:
            return priority_label
    return None

# ── LOAD ARNAIZ ──────────────────────────────────────────

print("Loading Arnaiz...")
arnaiz_path = r'data\raw\arnaiz-rodriguez\data\llm_label\gpt-4o-mini-labeled-sampled_dataset_n_2046_nPerD168_seed0-merged-labels.json'

with open(arnaiz_path, encoding='utf-8') as f:
    arnaiz_data = json.load(f)

all_records = []
skipped = 0

for item in arnaiz_data:
    raw = (item.get('label') or '').strip()
    if not raw or raw not in ARNAIZ_MAP:
        skipped += 1
        continue
    text = ' '.join(item['inputs']) if isinstance(item['inputs'], list) else str(item['inputs'])
    all_records.append({
        'text':           text[:2000],
        'label':          ARNAIZ_MAP[raw],
        'source':         'arnaiz',
        'original_label': raw,
    })

print(f"  Loaded: {len(all_records)}  Skipped: {skipped}")

# ── LOAD CRADLE ──────────────────────────────────────────

print("Loading CRADLE from HuggingFace...")
ds = load_dataset('SungJoo/Cradle-Bench')
split = list(ds.keys())[0]
print(f"  Split: {split}  Columns: {ds[split].column_names}")

# Find the text column automatically
text_col = None
for col in ds[split].column_names:
    sample = ds[split][0][col]
    if isinstance(sample, str) and len(sample) > 50:
        text_col = col
        break

print(f"  Text column detected: {text_col}")

cradle_added = 0
cradle_skipped = 0

for item in ds[split]:
    raw = item.get('final_labels', '')
    chosen = pick_cradle_label(raw)
    if not chosen or chosen not in CRADLE_MAP:
        cradle_skipped += 1
        continue

    text = item[text_col] if text_col else ''
    if isinstance(text, list):
        text = ' '.join(text)

    all_records.append({
        'text':           str(text)[:2000],
        'label':          CRADLE_MAP[chosen],
        'source':         'cradle',
        'original_label': raw,
    })
    cradle_added += 1

print(f"  Loaded: {cradle_added}  Skipped: {cradle_skipped}")

# ── DEDUPLICATE ──────────────────────────────────────────

print("Deduplicating...")
seen = set()
deduped = []
for r in all_records:
    key = r['text'][:200]
    if key not in seen:
        seen.add(key)
        deduped.append(r)

print(f"  Before: {len(all_records)}  After: {len(deduped)}")

# ── SAVE ────────────────────────────────────────────────

out_path = r'data\processed\crisis\merged_arnaiz_cradle.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ── REPORT ──────────────────────────────────────────────

print("\nFINAL DISTRIBUTION:")
dist = Counter(r['label'] for r in deduped)
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    bar = '█' * (count // 50)
    print(f"  {label:<22} {count:>5}  {bar}")

print(f"\nTotal saved: {len(deduped)}")
print(f"Output: {out_path}")