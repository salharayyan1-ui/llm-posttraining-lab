#!/usr/bin/env bash
# Run from the deployed repo. Optional PA2_DOWNLOAD_PROXY is a loopback SSH relay.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec > >(tee -a logs/setup.log) 2>&1
stage=starting
finish() {
    code=$?
    trap - EXIT
    /venv/main/bin/python - "$code" "$stage" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
Path('logs/setup_state.json').write_text(json.dumps({
    'updated_utc': datetime.now(timezone.utc).isoformat(),
    'exit_code': int(sys.argv[1]), 'last_stage': sys.argv[2],
    'status': 'passed' if sys.argv[1] == '0' else 'failed',
    'scope': 'dependency setup, objective tests, tiny-model CPU tests; cross-GPU test runs only when two GPUs are visible',
    'full_model_smoke': 'not_run', 'full_experiments': 'not_run',
}, indent=2))
PY
    printf '%s setup finished: exit=%s stage=%s\n' "$(date -u +%FT%TZ)" "$code" "$stage"
    exit "$code"
}
trap finish EXIT
run_stage() {
    stage="$1"
    shift
    printf '\n%s BEGIN %s\n' "$(date -u +%FT%TZ)" "$stage"
    printf 'Command: '; printf '%q ' "$@"; printf '\n'
    "$@"
    printf '%s PASS %s\n' "$(date -u +%FT%TZ)" "$stage"
}
if [[ -n "${PA2_DOWNLOAD_PROXY:-}" ]]; then
    export HTTPS_PROXY="$PA2_DOWNLOAD_PROXY" HTTP_PROXY="$PA2_DOWNLOAD_PROXY"
    export NO_PROXY=localhost,127.0.0.1
fi
export HF_HUB_DISABLE_XET=1
/venv/main/bin/python - <<'PY'
from pathlib import Path
import torch
Path('logs/torch.constraint.txt').write_text(f'torch=={torch.__version__}\n')
PY
run_stage dependencies uv pip install --python /venv/main/bin/python --index-url https://pypi.org/simple --constraint logs/torch.constraint.txt -r requirements.txt
run_stage environment /venv/main/bin/python -m scripts.check_environment
uv pip freeze --python /venv/main/bin/python > logs/environment.freeze.txt
run_stage objective_tests /venv/main/bin/python -m tests.test_objectives
run_stage model_execution_tests /venv/main/bin/python -m tests.test_model_execution
run_stage device_configs /venv/main/bin/python - <<'PY'
import torch
from common.data import load_yaml
from common.models import model_device
prefixes = ['configs', 'configs/smoke']
if torch.cuda.device_count() >= 2:
    prefixes += ['configs/dual_gpu', 'configs/dual_gpu/smoke']
for prefix in prefixes:
    for name in ['dpo', 'ppo', 'grpo', 'feedback']:
        cfg = load_yaml(f'{prefix}/{name}.yaml')
        assert str(model_device(cfg, 'policy')) == 'cuda:0'
        for role in ['reward', 'value', 'judge']:
            expected = 'cuda:1' if 'dual_gpu' in prefix else 'cuda:0'
            assert str(model_device(cfg, role)) == expected
        print(f'{prefix}/{name}.yaml: placement validated')
PY
stage=complete
