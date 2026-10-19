"""Full-model functional/memory checks; updates are discarded, not course results.

Each task runs in its own process. Logs go to logs/training_probe_<task>.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import time

import torch

from common.data import prompt_schedule, repo_path
from common.generation import scored_response_logprobs
from common.logging_utils import peak_vram, reset_peak_vram, save_json
from common.models import reference_mode, trainable_parameters


def synthetic_ids(tokenizer, count, row=0):
    ids = tokenizer.encode(f"Memory test row {row}: numbers 1 2 3 4 5. ", add_special_tokens=False)
    return (ids * (count // len(ids) + 1))[:count]


def synthetic_rollout(bundle, count, prompt_width, steps):
    device = next(bundle['policy'].parameters()).device
    tokenizer = bundle['tokenizer']
    sequences = torch.tensor([
        synthetic_ids(tokenizer, prompt_width, 0) +
        synthetic_ids(tokenizer, steps - 1, i + 1) + [tokenizer.eos_token_id]
        for i in range(count)
    ], device=device)
    return {'sequences': sequences, 'attention_mask': torch.ones_like(sequences),
            'prompt_width': prompt_width, 'response_ids': sequences[:, prompt_width:],
            'response_mask': torch.ones(count, steps, device=device),
            'truncated': [False] * count}


def timed(label, fn, records):
    reset_peak_vram()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    records.append({'name': label, 'wall_s': time.perf_counter() - start,
                    'result': result, **peak_vram()})
    print(label, records[-1], flush=True)


def check_stats(stats):
    for key, value in stats.items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise RuntimeError(f'Nonfinite {key}: {value}')


def probe_ppo(records):
    from task2_ppo.continue_train import prepare_ppo_continuation, collect_rollout, ppo_epoch, response_values, normalize
    from task2_ppo.ppo import shaped_rewards, compute_gae
    bundle = prepare_ppo_continuation('configs/ppo.yaml')
    cfg = bundle['cfg']
    rows = prompt_schedule(bundle['prompt_rows'], int(cfg['prompts_per_update']), int(cfg['seed']))

    def real_update():
        rollout = collect_rollout(bundle, rows, cfg)
        stats = [ppo_epoch(bundle, rollout, cfg) for _ in range(int(cfg['ppo_epochs']))]
        for s in stats: check_stats(s)
        return {'response_lengths': rollout['response_lengths'], 'truncated': rollout['truncated'], 'epochs': stats}
    timed('ppo_real_update', real_update, records)

    def max_update():
        gen = synthetic_rollout(bundle, int(cfg['prompts_per_update']), int(cfg['max_prompt_length']), int(cfg['max_response_length']))
        with torch.no_grad():
            old, _ = scored_response_logprobs(bundle['policy'], gen, with_entropy=False)
            with reference_mode(bundle['policy']):
                ref, _ = scored_response_logprobs(bundle['policy'], gen, with_entropy=False)
            values = response_values(bundle['value_model'], gen)
            rewards = shaped_rewards(torch.ones(old.shape[0], device=old.device), old, ref, gen['response_mask'], float(cfg['kl_beta']))
            advantages, returns = compute_gae(rewards, values, gen['response_mask'], float(cfg['gamma']), float(cfg['gae_lambda']))
            gen.update(old_logp=old, values=values, returns=returns,
                       advantages=normalize(advantages, gen['response_mask'], cfg['advantage_normalization']))
        stats = ppo_epoch(bundle, gen, cfg)
        check_stats(stats)
        return {'shape': list(gen['sequences'].shape), **stats}
    timed('ppo_max_length_backward', max_update, records)


def probe_grpo(records):
    from task3_grpo.continue_train import prepare_grpo_continuation, collect_groups, grpo_epoch
    from task3_grpo.grpo import group_relative_advantages
    bundle = prepare_grpo_continuation('configs/grpo.yaml')
    cfg = bundle['cfg']
    rows = prompt_schedule(bundle['prompt_rows'], int(cfg['prompts_per_update']), int(cfg['seed']))

    def real_update():
        groups = collect_groups(bundle, rows, cfg)
        stats, _ = grpo_epoch(bundle, groups, cfg, 'grpo')
        check_stats(stats)
        return {'response_lengths': groups['response_lengths'], 'truncated': groups['truncated'], **stats}
    timed('grpo_real_update', real_update, records)

    def max_update():
        k = int(cfg['num_generations'])
        count = int(cfg['prompts_per_update']) * k
        gen = synthetic_rollout(bundle, count, int(cfg['max_prompt_length']), int(cfg['max_completion_length']))
        with torch.no_grad():
            old, _ = scored_response_logprobs(bundle['policy'], gen, with_entropy=False)
            with reference_mode(bundle['policy']):
                ref, _ = scored_response_logprobs(bundle['policy'], gen, with_entropy=False)
            ids = torch.arange(count, device=old.device) // k
            rewards = torch.arange(count, device=old.device, dtype=torch.float32)
            gen.update(old_logp=old, ref_logp=ref, loss_mask=gen['response_mask'],
                       advantages=group_relative_advantages(rewards, ids))
        stats, grad_mass = grpo_epoch(bundle, gen, cfg, 'grpo')
        check_stats(stats)
        assert bool(torch.isfinite(grad_mass).all())
        assert float(grad_mass.sum()) > 0
        return {'shape': list(gen['sequences'].shape), 'token_gradient_mass': float(grad_mass.sum()), **stats}
    timed('grpo_max_length_backward', max_update, records)


def probe_dpo(records):
    from common.data import load_yaml, pad_batch
    from common.models import load_policy, load_tokenizer
    from task1_dpo.train import pair_logps
    from task1_dpo.dpo import dpo_loss
    cfg = load_yaml('configs/dpo.yaml')
    tokenizer = load_tokenizer(cfg['base_model'])
    policy = load_policy(cfg, trainable=True, fresh_lora=True)
    optimizer = torch.optim.AdamW(trainable_parameters(policy), lr=float(cfg['learning_rate']))
    length = int(cfg['max_sequence_length'])
    prompt_width = int(cfg['max_prompt_length'])
    def max_update():
        def examples(offset):
            return [(synthetic_ids(tokenizer, prompt_width, 0) +
                     synthetic_ids(tokenizer, length - prompt_width - 1, offset + i) + [tokenizer.eos_token_id],
                     [0] * prompt_width + [1] * (length - prompt_width))
                    for i in range(int(cfg['batch_size']))]
        batch = {'chosen': pad_batch(tokenizer, examples(0)), 'rejected': pad_batch(tokenizer, examples(5))}
        optimizer.zero_grad(set_to_none=True)
        loss, _ = dpo_loss(*pair_logps(policy, batch), float(cfg['beta']))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg['max_grad_norm']))
        optimizer.step()
        result = {'pairs': int(cfg['batch_size']), 'sequence_length': length, 'loss': float(loss), 'grad_norm': float(norm)}
        check_stats(result)
        return result
    timed('dpo_max_length_backward', max_update, records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['dpo', 'ppo', 'grpo'], required=True)
    args = ap.parse_args()
    records = []
    state = {'task': args.task, 'scope': 'discarded functional and maximum-shape training probes; not experimental results',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'records': records}
    try:
        {'dpo': probe_dpo, 'ppo': probe_ppo, 'grpo': probe_grpo}[args.task](records)
        state['status'] = 'passed'
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(repo_path(f'logs/training_probe_{args.task}.json'), state)


if __name__ == '__main__':
    main()
