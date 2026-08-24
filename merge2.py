# ============================================================
# TAXONOMY NOTE — other_violence DROPPED (August 2026)
# ============================================================
# Original taxonomy included other_violence to cover violent
# thoughts, child abuse, and general interpersonal violence.
#
# After merging Arnaiz + CRADLE with correct priority-based
# label assignment, other_violence collapsed to only 46 examples.
# Root cause: most CRADLE records that appeared to be
# other_violence (child abuse) also carried suicidal ideation
# or self-harm labels — and those correctly win the priority
# tie-break. The 46 remaining are genuinely other_violence only.
#
# 46 examples → ~32 training examples after 70/15/15 split.
# This is not enough to train a reliable classifier.
#
# DECISION: Drop other_violence for this iteration.
# Final taxonomy is 6 categories (no_crisis + 5 crisis types).
#
# FUTURE WORK: Augment other_violence with HuggingFace datasets
# targeting violent thoughts and child abuse content, then
# reintroduce as a seventh category.
# ============================================================
import json, os
from datasets import load_dataset
from collections import Counter

os.makedirs('data/processed/crisis', exist_ok=True)

# ── MASTER MAPS ──────────────────────────────────────────
ARNAIZ_MAP = {
    'suicidal_ideation':             ('suicidal_ideation',  'suicidal',        'unknown', 'mapped'),
    'self_harm':                     ('self_directed_harm', 'self_harm',       'unknown', 'mapped'),
    'risk_taking_behaviours':        ('self_directed_harm', 'risk_taking',     'unknown', 'mapped'),
    'risk_taking':                   ('self_directed_harm', 'risk_taking',     'unknown', 'mapped'),
# other_violence removed — see taxonomy note at top of file
# 'childabuse_endangerment_ongoing': dropped
# 'childabuse_endangerment_past':    dropped
# 'violent_thoughts':                dropped
# 'violence':                        dropped
    'anxiety_crisis':                ('acute_crisis',       'anxiety',         'unknown', 'mapped'),
    'substance_abuse_or_withdrawal': ('acute_crisis',       'substance',       'unknown', 'mapped'),
    'substance_abuse':               ('acute_crisis',       'substance',       'unknown', 'mapped'),
    'no_crisis':                     ('no_crisis',          'none',            'unknown', 'mapped'),
}

CRADLE_MAP = {
    'suicideideation_active_ongoing':  ('suicidal_ideation',  'active',        'ongoing', 'mapped'),
    'suicideideation_passive_ongoing': ('suicidal_ideation',  'passive',       'ongoing', 'mapped'),
    'suicideideation_active_past':     ('suicidal_ideation',  'active',        'past',    'mapped'),
    'suicideideation_passive_past':    ('suicidal_ideation',  'passive',       'past',    'mapped'),
    'selfharm_ongoing':                ('self_directed_harm', 'self_harm',     'ongoing', 'mapped'),
    'selfharm_past':                   ('self_directed_harm', 'self_harm',     'past',    'mapped'),
    'domesticviolence_ongoing':        ('domestic_violence',  'partner_abuse', 'ongoing', 'mapped'),
    'domesticviolence_past':           ('domestic_violence',  'partner_abuse', 'past',    'mapped'),
    'rape_ongoing':                    ('sexual_violence',    'rape',          'ongoing', 'mapped'),
    'rape_past':                       ('sexual_violence',    'rape',          'past',    'mapped'),
    'sexualharassment_ongoing':        ('sexual_violence',    'harassment',    'ongoing', 'mapped'),
    'sexualharassment_past':           ('sexual_violence',    'harassment',    'past',    'mapped'),
# other_violence removed — see taxonomy note at top of file
# 'childabuse_endangerment_ongoing': dropped
# 'childabuse_endangerment_past':    dropped
# 'violent_thoughts':                dropped
# 'violence':                        dropped
    'no_crisis':                       ('no_crisis',          'none',          'unknown', 'mapped'),
}

all_records = []

# ── SOURCE 1: ARNAIZ ─────────────────────────────────────
print('Loading Arnaiz...')
arnaiz_path = 'data/raw/arnaiz-rodriguez/data/llm_label/gpt-4o-mini-labeled-sampled_dataset_n_2046_nPerD168_seed0-merged-labels.json'
with open(arnaiz_path, encoding='utf-8') as f:
    arnaiz_data = json.load(f)

for item in arnaiz_data:
    raw = item.get('label', '')
    if not raw or raw not in ARNAIZ_MAP:
        continue
    cat, subtype, temporal, method = ARNAIZ_MAP[raw]
    text = ' '.join(item['inputs']) if isinstance(item['inputs'], list) else str(item['inputs'])
    all_records.append({
        'text':           text[:2000],
        'label':          cat,
        'subtype':        subtype,
        'temporal':       temporal,
        'source_dataset': 'arnaiz',
        'label_method':   method,
        'confidence':     1.0,
        'original_label': raw,
    })

print(f'  Arnaiz: {len(all_records)} records')

# ── SOURCE 2: CRADLE ─────────────────────────────────────
print('Loading CRADLE...')
ds = load_dataset('SungJoo/Cradle-Bench')
split = list(ds.keys())[0]

# Find text column
text_col = None
for col in ds[split].column_names:
    sample = ds[split][0][col]
    if isinstance(sample, str) and len(sample) > 50:
        text_col = col
        break
    elif isinstance(sample, list) and sample and isinstance(sample[0], str):
        text_col = col
        break

print(f'  Text column: {text_col}')
cradle_added = 0
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
    'no_crisis',
]
for item in ds[split]:
    raw = item['final_labels']
    parts = [p.strip() for p in raw.split(',')] if isinstance(raw, str) else raw

    # Most severe label wins — not first in string
    matched = None
    for priority_label in CRADLE_PRIORITY:
        if priority_label in parts:
            matched = (priority_label, CRADLE_MAP[priority_label])
            break

    if not matched:
        continue

    raw_label, (cat, subtype, temporal, method) = matched

    text = item[text_col] if text_col else ''
    if isinstance(text, list):
        text = ' '.join(text)

    all_records.append({
        'text':           str(text)[:2000],
        'label':          cat,
        'subtype':        subtype,
        'temporal':       temporal,
        'source_dataset': 'cradle',
        'label_method':   method,
        'confidence':     1.0,
        'original_label': raw,
    })
    cradle_added += 1

print(f'  CRADLE: {cradle_added} records')

# ── DEDUP ────────────────────────────────────────────────
print('Deduplicating...')
seen = set()
deduped = []
for r in all_records:
    key = r['text'][:200]
    if key not in seen:
        seen.add(key)
        deduped.append(r)

print(f'  Before dedup: {len(all_records)}')
print(f'  After dedup:  {len(deduped)}')

# ── SAVE ─────────────────────────────────────────────────
out_path = 'data/processed/crisis/merged_arnaiz_cradle.jsonl'
with open(out_path, 'w', encoding='utf-8') as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ── REPORT ───────────────────────────────────────────────
print()
print('FINAL DISTRIBUTION:')
dist = Counter(r['label'] for r in deduped)
target = 400
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    bar = '█' * (count // 50)
    gap = max(0, target - count)
    status = '✅' if count >= target else '⚠️ '
    print(f'  {status} {label:<22} {count:>5}  {bar}')
    if gap:
        print(f'     └─ needs {gap} more from HF mining')

print(f'\nTotal saved: {len(deduped)}')
print(f'Output: {out_path}')
