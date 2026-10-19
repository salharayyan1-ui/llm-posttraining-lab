"""Independently reconcile saved records and summaries without Torch or model calls.

Run from the repository root with a standard-library Python interpreter.
This checks numerical bookkeeping, not human judgments or scientific conclusions.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='results/reproducibility/results_consistency_check.json',
                    help='Report path within the repository; use a dated path to preserve earlier audits.')
args = parser.parse_args()
destination = (ROOT / args.output).resolve()
if not destination.is_relative_to(ROOT):
    parser.error('--output must stay within the repository')
checks, failures, sources = [], [], {}


def read(path, lines=False, table=False):
    p = ROOT / path
    raw = p.read_bytes()
    sources[str(p.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8-sig')
    if lines:
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    if table:
        return list(csv.DictReader(text.splitlines()))
    return json.loads(text)


def check(label, expected, actual, abs_tol=1e-6):
    if isinstance(expected, dict):
        for k, v in expected.items():
            check(label + '/' + k, v, actual.get(k), abs_tol)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        ok = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-6, abs_tol=abs_tol)
    else:
        ok = expected == actual
    checks.append(label)
    if not ok:
        failures.append({'check': label, 'expected': expected, 'actual': actual})


def mean(x):
    return statistics.fmean(list(x))


def quantile(xs, p):
    a = sorted(xs)
    i = (len(a) - 1) * p
    lo = math.floor(i)
    return a[lo] + (a[math.ceil(i)] - a[lo]) * (i - lo)


def lengths(xs):
    xs = list(xs)
    q1, q3 = quantile(xs, .25), quantile(xs, .75)
    return dict(n=len(xs), mean=mean(xs), std=statistics.pstdev(xs), median=quantile(xs, .5), q1=q1, q3=q3, iqr=q3-q1)


def generations(rows):
    total = sum(r['response_tokens'] for r in rows)
    return dict(n_prompts=len(rows), reward_mean=mean(r['reward'] for r in rows),
                reward_std=statistics.pstdev(r['reward'] for r in rows),
                kl_token_mean=sum(r['kl_sum'] for r in rows)/total,
                kl_sequence_mean=mean(r['kl_sum'] for r in rows),
                entropy_token_mean=sum(r['entropy_sum'] for r in rows)/total,
                response_length=lengths(r['response_tokens'] for r in rows),
                eos_rate=mean(r['terminated_with_eos'] for r in rows),
                truncation_rate=mean(r['truncated'] for r in rows))


def pair_stats(rows, beta):
    def softplus(z):
        return max(z, 0) + math.log1p(math.exp(-abs(z)))
    return dict(n_pairs=len(rows), dpo_loss=mean(softplus(-beta*r['margin']) for r in rows),
                preference_accuracy=mean(r['margin'] > 0 for r in rows),
                margin_mean=mean(r['margin'] for r in rows),
                chosen_reward_mean=mean(beta*(r['policy_chosen']-r['ref_chosen']) for r in rows),
                rejected_reward_mean=mean(beta*(r['policy_rejected']-r['ref_rejected']) for r in rows))


for task in ('task1_dpo', 'task2_ppo', 'task3_grpo'):
    protocol = None
    for folder in sorted((ROOT/'results'/task/'eval').iterdir()):
        if not folder.is_dir():
            continue
        rel = folder.relative_to(ROOT).as_posix()
        rows = read(rel+'/generations.jsonl', lines=True)
        m = read(rel+'/metrics.json')
        check(rel+'/generation', generations(rows), m['generation'] if task=='task1_dpo' else m)
        current = [(r['prompt_id'], r['effective_prompt_messages'], r['effective_generation_config']) for r in rows]
        if protocol is None:
            protocol = current
        check(rel+'/matched_prompt_and_decoding_protocol', True, current == protocol)
        cap = rows[0]['effective_generation_config']['max_new_tokens']
        check(rel+'/token_lengths_valid', True, all(0 < r['response_tokens'] <= cap for r in rows))
        check(rel+'/EOS_and_truncation_consistent', True, all(r['terminated_with_eos'] != r['truncated'] for r in rows))
        if task == 'task1_dpo':
            for fname, field in [('pairs_standard.jsonl', 'heldout_pairs'), ('pairs_stratified.jsonl', 'stratified_pairs')]:
                pairs = read(rel+'/'+fname, lines=True)
                for i, r in enumerate(pairs):
                    check(rel+'/'+fname+f'/margin/{i}', (r['policy_chosen']-r['policy_rejected'])-(r['ref_chosen']-r['ref_rejected']), r['margin'], 5e-4)
                expected = pair_stats(pairs, m['beta'])
                check(rel+'/'+field, expected, m[field]['all'] if field=='stratified_pairs' else m[field], 5e-5)
                if field=='stratified_pairs':
                    for stratum in sorted({r['length_stratum'] for r in pairs}):
                        check(rel+'/'+stratum, pair_stats([r for r in pairs if r['length_stratum']==stratum], m['beta']), m[field]['by_stratum'][stratum], 5e-5)
            word = read(rel+'/word_limit.jsonl', lines=True)
            # Word counts are checked directly against response text below; field names are saved by the evaluator.
            for i, r in enumerate(word):
                check(rel+f'/word_count/{i}', len(re.findall(r'\b\w+\b', r['response'])), r['words'])
                check(rel+f'/word_compliance/{i}', float(r['words'] <= r['word_limit']), r['compliant'])
            check(rel+'/word_limit_summary', dict(n_responses=len(word), compliance_rate=mean(r['compliant'] for r in word),
                  words=lengths(r['words'] for r in word), words_over_limit_ratio_mean=mean(r['words']/r['word_limit'] for r in word),
                  response_tokens=lengths(r['response_tokens'] for r in word)), m['word_limit'])


fork_tokens = {}
for task, name in [('task2_ppo','rollouts.jsonl'), ('task3_grpo','completions.jsonl')]:
    for folder in sorted((ROOT/'results'/task/'train').iterdir()):
        if not folder.is_dir():
            continue
        rel = folder.relative_to(ROOT).as_posix()
        s = read(rel+'/summary.json')
        rows = read(rel+'/'+name, lines=True)
        k = s['config'].get('num_generations', 1) if task=='task3_grpo' else 1
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r['update']].append(r)
        check(rel+'/updates', s['updates'], len(groups))
        check(rel+'/group_sizes', True, all(len(g)==k for g in groups.values()))
        check(rel+'/generated_tokens', sum(r['response_tokens'] for r in rows), s['generated_tokens'])
        check(rel+'/prompt_schedule', s['prompt_ids'], [g[0]['prompt_id'] for g in groups.values()])
        cap = s['effective_generation_config']['max_new_tokens']
        check(rel+'/maximum_budget', s['updates']*k*cap, s['maximum_generated_token_budget'])
        check(rel+'/tokens_within_cap', True, all(0 < r['response_tokens'] <= cap for r in rows))
        if task=='task3_grpo':
            check(rel+'/truncation_mask', True, all(not r['in_loss'] if r['truncated'] else r['in_loss'] for r in rows))
        if s['updates']==8:
            fork_tokens[task+'/'+folder.name] = s['generated_tokens']


safety = read('results/task4_safety/safety_rates.csv', table=True)
contradictions = {}
for saved in safety:
    policy = saved['policy']
    rows = read(f'results/task4_safety/judged_{policy}.jsonl', lines=True)
    safe = [r for r in rows if r['benchmark_class']=='SAFE']
    unsafe = [r for r in rows if r['benchmark_class']=='UNSAFE']
    rate = lambda rs, labels: mean(r['judge_label'] in labels for r in rs)
    expected = dict(n_safe=len(safe), n_unsafe=len(unsafe),
        safe_answer_rate=rate(safe, ['SAFE_ANSWER']), over_refusal_rate=rate(safe, ['OVER_REFUSAL']),
        unsafe_compliance_rate=rate(unsafe, ['UNSAFE_COMPLIANCE']), justified_refusal_rate=rate(unsafe, ['JUSTIFIED_REFUSAL']),
        ambiguous_rate=rate(rows, ['AMBIGUOUS']), ambiguous_rate_safe=rate(safe, ['AMBIGUOUS']), ambiguous_rate_unsafe=rate(unsafe, ['AMBIGUOUS']),
        safe_labeled_as_unsafe_request=rate(safe, ['JUSTIFIED_REFUSAL','UNSAFE_COMPLIANCE']),
        unsafe_labeled_as_safe_request=rate(unsafe, ['SAFE_ANSWER','OVER_REFUSAL']),
        response_tokens_mean=mean(r['response_tokens'] for r in rows), response_tokens_std=statistics.pstdev(r['response_tokens'] for r in rows),
        response_tokens_mean_safe=mean(r['response_tokens'] for r in safe), response_tokens_mean_unsafe=mean(r['response_tokens'] for r in unsafe))
    check('safety/'+policy, expected, {k:float(saved[k]) for k in expected})
    contradictions[policy] = {k:expected[k] for k in ['safe_labeled_as_unsafe_request','unsafe_labeled_as_safe_request']}


math_judge = {}
final_re = re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)')
for dataset in ['gsm','transfer']:
    rel = 'results/task5_feedback/'+dataset
    m = read(rel+'/metrics.json')
    by_policy = {}
    for policy in ['sft','rlvr','rlaif']:
        rows = read(rel+'/generated_'+policy+'.jsonl', lines=True)
        by_policy[policy] = {r['source_index']:r for r in rows}
        for i,r in enumerate(rows):
            matches = final_re.findall(r['response'])
            pred = matches[-1].replace(',','') if matches else None
            correct = float(pred is not None and abs(float(pred)-float(r['gold_final']))<=1e-9)
            ft = 'correct' if correct else 'wrong_final' if pred is not None else 'truncated' if r['truncated'] else 'no_final'
            check(rel+'/'+policy+f'/scoring/{i}', dict(pred_final=pred, format_ok=pred is not None, correct=correct, failure_type=ft), r)
        expected = dict(n=len(rows), exact_accuracy=mean(r['correct'] for r in rows), format_compliance=mean(r['format_ok'] for r in rows),
                        truncation_rate=mean(r['truncated'] for r in rows), response_tokens=lengths(r['response_tokens'] for r in rows),
                        failure_types=dict(collections.Counter(r['failure_type'] for r in rows)))
        check(rel+'/'+policy+'/summary', expected, m['policies'][policy])
    pairs = read(rel+'/pairwise.jsonl', lines=True)
    for r in pairs:
        a,b = by_policy[r['policy']][r['source_index']], by_policy['sft'][r['source_index']]
        check(rel+'/pair_source/'+r['policy']+'/'+str(r['source_index']),
              dict(policy_correct=a['correct'], sft_correct=b['correct'], policy_tokens=a['response_tokens'], sft_tokens=b['response_tokens'],
                   verifier='A' if a['correct']>b['correct'] else 'B' if a['correct']<b['correct'] else 'TIE'), r)
    for policy in ['rlvr','rlaif']:
        rs = [r for r in pairs if r['policy']==policy]
        decisive = [r for r in rs if r['verifier']!='TIE']
        tied = [r for r in rs if r['verifier']=='TIE']
        fresh = [r for r in rs if not r['judge_cache_hit']]
        expected = dict(n=len(rs), wins=sum(r['judge']=='A' for r in rs), ties=sum(r['judge']=='TIE' for r in rs), losses=sum(r['judge']=='B' for r in rs),
                        win_rate=mean(1 if r['judge']=='A' else .5 if r['judge']=='TIE' else 0 for r in rs),
                        verifier_judge_agreement=mean(r['judge']==r['verifier'] for r in rs),
                        agreement_when_verifier_decisive=mean(r['judge']==r['verifier'] for r in decisive),
                        judge_decisive_when_verifier_ties=mean(r['judge']!='TIE' for r in tied),
                        verifier_decisive_rate=len(decisive)/len(rs), judge_seconds_per_comparison=mean(r['judge_seconds'] for r in fresh))
        check(rel+'/'+policy+'/pair_summary', expected, m['pairwise_vs_sft'][policy])
        math_judge[dataset+'/'+policy] = dict(tie_rate=expected['ties']/len(rs), agreement_when_verifier_decisive=expected['agreement_when_verifier_decisive'])


rel = 'results/task5_feedback/diagnostics'
m = read(rel+'/summary.json')
pairs = read(rel+'/pairs.jsonl', lines=True)
variants = read(rel+'/variant_scores.csv', table=True)
check(rel+'/problem_count', len({r['problem_id'] for r in pairs}), m['n_problems'])
check(rel+'/order_consistency', mean(r['rlaif']==r['rlaif_reversed'] for r in pairs), m['judge_order_consistency'])
check(rel+'/verifier_mismatches', sum(float(r['rlvr_reward'])!=float(r['expected_exact_reward']) for r in variants), m['verifier_vs_expected_mismatches'])
for mechanism in ['rlvr','rlaif','rlaif_reversed']:
    reason = [r for r in pairs if r['pair']=='reasoning']
    outcome = [r for r in pairs if r['pair'] in ['outcome','distractor']]
    expected = dict(S_reason=mean(r[mechanism]=='better' for r in reason), reason_tie_rate=mean(r[mechanism]=='tie' for r in reason), reason_wrong_rate=mean(r[mechanism]=='wrong' for r in reason),
                    S_outcome=mean(r[mechanism]=='better' for r in outcome), outcome_tie_rate=mean(r[mechanism]=='tie' for r in outcome), outcome_wrong_rate=mean(r[mechanism]=='wrong' for r in outcome))
    check(rel+'/'+mechanism, expected, m['sensitivity'][mechanism])
for variant in {r['variant'] for r in variants}:
    rs = [r for r in variants if r['variant']==variant]
    check(rel+'/'+variant, {k:mean(float(r[k]) for r in rs) for k in ['rlvr_reward','rlaif_round_robin']}, m['variant_means'][variant])


out = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), status='passed_with_known_limitations' if not failures else 'failed',
           checks=len(checks), failures=failures, source_hashes=sources, actual_short_fork_tokens=fork_tokens,
           safety_label_contradictions=contradictions, math_judge_diagnostics=math_judge,
           limits=['No model likelihood or tensor recomputation: no local Torch installed.',
                   'GRPO normalization forks have unequal actual emitted-token totals (4793 vs 4914).',
                   'Safety judge contradicts benchmark request classes; blind human audit remains required.',
                   'Pairwise judge collapses unparsable output into TIE; raw judge text was not saved, so valid ties cannot be separated from parse failures.',
                   'This is a numerical bookkeeping audit, not human-authored coursework interpretation.'])
dest = destination
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:out[k] for k in ['status','checks','failures','actual_short_fork_tokens']}, indent=2))
sys.exit(bool(failures))
