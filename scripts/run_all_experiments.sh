#!/bin/bash
# SkillSieve Full Experiment Runner
# Run on Orange Pi after setup and data transfer
set -e

cd /root/skillsieve

# Load API keys
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

SKILLS_DIR="data/raw/openclaw-skills/skills"
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "=== SkillSieve Experiment Suite ==="
echo "Skills directory: $SKILLS_DIR"
echo ""

# Step 1: Layer 1 full scan (zero API cost)
echo "[Step 1/5] Layer 1 static scan on all skills..."
python3 experiments/real_scan.py \
    --skills-dir $SKILLS_DIR \
    --output $RESULTS_DIR/exp1_layer1_scan.json

echo ""

# Step 2: Build labels from Layer 1 results + known malicious authors
echo "[Step 2/5] Building labels from scan results..."
python3 -c "
import json
from pathlib import Path

# Load Layer 1 scan results
scan = json.loads(Path('$RESULTS_DIR/exp1_layer1_scan.json').read_text())

# Known malicious authors from ToxicSkills + ClawHavoc research
KNOWN_MALICIOUS_AUTHORS = {
    'hightower6eu',   # 250+ typosquatting skills (VirusTotal confirmed)
    'moonshine-100rze',  # moltbook malware (ToxicSkills referenced)
}

# Build labels: known malicious + high-confidence L1 flags
labels = {}
for skill in scan.get('top_50_flagged_skills', []):
    path = skill['path']
    author = path.split('/')[0] if '/' in path else 'unknown'
    name = path.split('/')[-1] if '/' in path else path
    if author in KNOWN_MALICIOUS_AUTHORS or skill['score'] > 0.8:
        labels[name] = 'malicious'
    else:
        labels[name] = 'benign'

Path('$RESULTS_DIR/labels_auto.json').write_text(json.dumps(labels, indent=2))
print(f'Generated {len(labels)} labels ({sum(1 for v in labels.values() if v==\"malicious\")} malicious)')
"

echo ""

# Step 3: Generate adversarial samples
echo "[Step 3/5] Generating adversarial test samples..."
python3 -c "
from pathlib import Path
from dataset.adversarial.generator import AdversarialGenerator

gen = AdversarialGenerator()
samples = gen.generate_all()
out = Path('data/adversarial')
gen.save_samples(samples, out)
print(f'Generated {len(samples)} adversarial samples in {out}')
"

echo ""

# Step 4: Layer 2 scan on suspicious skills (requires API key)
echo "[Step 4/5] Layer 2 LLM scan on suspicious skills..."
if [ -z "$KIMI_API_KEY" ]; then
    echo "  SKIPPED: KIMI_API_KEY not set in .env"
else
    python3 -c "
import asyncio, json, time
from pathlib import Path
from skillsieve.core.pipeline import Pipeline

# Load suspicious skills from Layer 1 results
scan = json.loads(Path('$RESULTS_DIR/exp1_layer1_scan.json').read_text())
flagged = scan.get('top_50_flagged_skills', [])
skills_dir = Path('$SKILLS_DIR')

pipeline = Pipeline(max_layer=2)
results = []
for i, skill in enumerate(flagged[:50]):  # Limit to top 50 for cost control
    path = skills_dir / skill['path']
    if not path.exists():
        continue
    print(f'  [{i+1}/{min(len(flagged),50)}] Scanning {skill[\"path\"]}...')
    try:
        result = asyncio.run(pipeline.scan(path))
        results.append({
            'path': skill['path'],
            'verdict': result.final_verdict.value,
            'confidence': result.final_confidence,
            'layer_stopped': result.layer_stopped,
        })
    except Exception as e:
        print(f'    Error: {e}')
        results.append({'path': skill['path'], 'error': str(e)})

Path('$RESULTS_DIR/exp4_layer2_scan.json').write_text(json.dumps(results, indent=2))
print(f'Layer 2 scan complete: {len(results)} skills processed')
"
fi

echo ""

# Step 5: Summary
echo "[Step 5/5] Generating summary..."
python3 -c "
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
print('=== EXPERIMENT RESULTS SUMMARY ===')
print()

# Layer 1
if (results_dir / 'exp1_layer1_scan.json').exists():
    l1 = json.loads((results_dir / 'exp1_layer1_scan.json').read_text())
    s = l1['summary']
    print(f'Layer 1 Static Scan:')
    print(f'  Total scanned: {s[\"total_scanned\"]}')
    print(f'  Safe: {s[\"safe\"]}')
    print(f'  Suspicious: {s[\"suspicious\"]} ({s[\"suspicious_rate\"]}%)')
    print(f'  Errors: {s[\"errors\"]}')
    if l1.get('latency'):
        print(f'  Avg latency: {l1[\"latency\"][\"avg_ms\"]:.1f}ms/skill')
    print()

# Layer 2
if (results_dir / 'exp4_layer2_scan.json').exists():
    l2 = json.loads((results_dir / 'exp4_layer2_scan.json').read_text())
    mal = sum(1 for r in l2 if r.get('verdict') == 'malicious')
    safe = sum(1 for r in l2 if r.get('verdict') == 'safe')
    err = sum(1 for r in l2 if 'error' in r)
    print(f'Layer 2 LLM Scan (top 50 suspicious):')
    print(f'  Malicious: {mal}')
    print(f'  Safe: {safe}')
    print(f'  Errors: {err}')
    print()

print('Results saved to: $RESULTS_DIR/')
"

echo ""
echo "=== All experiments complete ==="
