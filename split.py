import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split

random.seed(42)

# ── LOAD ─────────────────────────────────────────────────
with open('data/processed/crisis/merged_arnaiz_cradle.jsonl', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

print(f"Total records: {len(all_data)}")

# ── SPLIT FIRST — on full unmodified data ─────────────────
labels = [r['label'] for r in all_data]

train_full, temp = train_test_split(
    all_data, test_size=0.30,
    stratify=labels, random_state=42
)
temp_labels = [r['label'] for r in temp]
val, test = train_test_split(
    temp, test_size=0.50,
    stratify=temp_labels, random_state=42
)

print(f"Train (pre-cap): {len(train_full)}")
print(f"Val:             {len(val)}")
print(f"Test:            {len(test)}")

# ── CAP no_crisis IN TRAIN ONLY — after split ─────────────
NO_CRISIS_CAP = 800

no_crisis = [r for r in train_full if r['label'] == 'no_crisis']
other     = [r for r in train_full if r['label'] != 'no_crisis']

if len(no_crisis) > NO_CRISIS_CAP:
    no_crisis = random.sample(no_crisis, NO_CRISIS_CAP)
    print(f"no_crisis capped at {NO_CRISIS_CAP} in training")

train = other + no_crisis
random.shuffle(train)

# ── REPORT ───────────────────────────────────────────────
print("\nTRAIN distribution (post-cap):")
dist = Counter(r['label'] for r in train)
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {label:<22} {count:>5}")

print("\nVAL distribution (natural):")
dist = Counter(r['label'] for r in val)
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {label:<22} {count:>5}")

print("\nTEST distribution (natural):")
dist = Counter(r['label'] for r in test)
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    print(f"  {label:<22} {count:>5}")

# ── SAVE ─────────────────────────────────────────────────
for name, split_data in [('train', train), ('val', val), ('test', test)]:
    path = f'data/processed/crisis/{name}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for r in split_data:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nSaved {name}.jsonl — {len(split_data)} records")