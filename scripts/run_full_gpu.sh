#!/usr/bin/env bash
# Supervised full experiment job. Outputs are persistent; coursework report is human work.
set -euo pipefail
cd /workspace/pa2-llm-posttraining
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

record_state() {
  /venv/main/bin/python - "$1" "$2" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
p = Path('logs/full_run_state.json')
old = json.loads(p.read_text()) if p.exists() else {}
now = datetime.now(timezone.utc).isoformat()
state = {'status': sys.argv[1], 'return_code': int(sys.argv[2]),
         'started_utc': old.get('started_utc', now), 'updated_utc': now,
         'scope': 'full experimental pipeline; human audit/report and exact actual-token budget caveat remain',
         'command': 'python -m scripts.run_pipeline; python -m scripts.verify_pipeline_outputs'}
tmp = p.with_suffix('.json.tmp')
tmp.write_text(json.dumps(state, indent=2))
tmp.replace(p)
PY
}

finish() {
  code=$?
  trap - EXIT
  if [ "$code" -eq 0 ]; then record_state passed "$code"; else record_state failed "$code"; fi
  date -u "+%Y-%m-%dT%H:%M:%SZ FULL_RUN_EXIT=$code"
  exit "$code"
}
trap finish EXIT
record_state running 0
date -u '+%Y-%m-%dT%H:%M:%SZ BEGIN_FULL_PIPELINE'
timeout --signal=TERM --kill-after=30s 710m /venv/main/bin/python -u -m scripts.run_pipeline
/venv/main/bin/python -u -m scripts.verify_pipeline_outputs
