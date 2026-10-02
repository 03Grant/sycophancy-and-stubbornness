"""One entry point for the CoPE-Bench comparison: run any subset of the arms by name, or all of them, then score.

    python run.py --model Qwen/Qwen3.5-9B --methods all
    python run.py --model Qwen/Qwen3.5-9B --methods original,spae,autopasta --limit 20

Arms: original, spae, random, oracle (the two-call runner) and s2a, cad, adacad, caa, juice, autopasta (the baselines).
`all` is the paper's main table: original, the six baselines and spae. The Original arm is always run first because its
neutral rows define the eligibility gate; the auxiliary call runs once and is shared by spae / random / oracle. The fitted
baselines (caa, juice, autopasta) read the artifacts that fit.py wrote for the backbone from out/<label>/artifacts/. With
--seeds, every requested arm is rerun with the reply sampled at the backbone's shipped values (the sampled-decoding
appendix) and aggregated over the seeds. --device-map auto spreads a backbone over every visible GPU.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORED = 'wrong_claim,correct_claim,context_conflict,context_consistent,wrong_claim_conflict'
ALL = ['original', 's2a', 'cad', 'adacad', 'caa', 'juice', 'autopasta', 'spae']
CONTROLS = ['random', 'oracle']
TWO_CALL = {'original', 'spae', 'random', 'oracle'}

# the paper's frozen settings per backbone (Table of SPAE parameters) and each backbone's shipped sampling values
BACKBONES = {
    'Qwen3.5-4B': {'label': 'qwen35_4b', 'spae': '--alpha-memory 0.4 --lin-alpha 0.4 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.9', 'think': True, 'sampling': (0.7, 0.8, 20)},
    'Qwen3.5-9B': {'label': 'qwen35_9b', 'spae': '--alpha-memory 0.3 --lin-alpha 0.3 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.5', 'think': True, 'sampling': (0.7, 0.8, 20)},
    'Qwen2.5-14B-Instruct': {'label': 'qwen25_14b', 'spae': '--alpha-memory 0.1 --max-transfer 0.25 --target-share 0.5', 'think': False, 'sampling': (0.7, 0.8, 20)},
    'gemma-4-26B-A4B-it': {'label': 'gemma4_26b', 'spae': '--alpha-memory 0.3 --max-transfer 0.5 --target-share 0.7 --layers 3:30:4', 'think': False, 'sampling': (1.0, 0.95, 64)},
    'Qwen3.8-27B': {'label': 'qwen38_27b', 'spae': '--alpha-memory 0.3 --lin-alpha 0.3 --max-transfer 0.5 --target-share 0.7 --lin-rho 0.5', 'think': True, 'sampling': (1.0, 0.95, 20)},
}
SPAE_COMMON = '--kernel share --receivers-memory all --receivers-context after --localiser quote --full-letters'


def backbone(model):
    """The settings entry whose name is a suffix of the model id or path (case-insensitive), else None."""
    low = model.lower().rstrip('/')
    for name, cfg in BACKBONES.items():
        if low.endswith(name.lower()):
            return cfg
    return None


def run(cmd, log):
    """Run one stage, teeing its output to a log file; abort on failure."""
    print('+', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    with open(log, 'a') as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    if proc.returncode:
        sys.exit(f'stage failed (see {log})')


def main():
    """Resolve the arms, run the stages in dependency order and score."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Hugging Face id or local path of the backbone')
    p.add_argument('--methods', default='all', help="comma list of arms, or 'all'")
    p.add_argument('--data', default=str(HERE.parent / 'CoPE-Bench' / 'cope_bench_test.jsonl'))
    p.add_argument('--out', help='output directory (default out/<label>)')
    p.add_argument('--label', help='backbone label for the output directory and the artifacts (default: from the settings table, else the model name)')
    p.add_argument('--artifacts', help='fitted-baseline artifacts directory, as written by fit.py (default out/<label>/artifacts)')
    p.add_argument('--variant', default='fit-mix', choices=['fit-mix', 'fit-syco', 'fit-stub'], help='which fit of CAA / JuICE / AutoPASTA')
    p.add_argument('--spae-flags', help='override the SPAE parameters (default: the settings table entry of the backbone)')
    p.add_argument('--no-think', action='store_true', help='force the thinking switch off (default: from the settings table)')
    p.add_argument('--device-map', help='spread the model over several GPUs with accelerate, e.g. auto (default: one device)')
    p.add_argument('--batch', type=int, default=1, help='batch size of the baseline runners')
    p.add_argument('--limit', type=int, help='smoke test: rows per stage')
    p.add_argument('--rows', help='row-id file (one per line): every stage runs on these rows only')
    p.add_argument('--no-original', action='store_true', help='do not add the Original arm (for splitting the arms over several processes); scoring then needs original.jsonl from another run')
    p.add_argument('--seeds', nargs='*', type=int, help='sampled decoding: rerun the arms with these seeds and aggregate')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--skip-score', action='store_true')
    a = p.parse_args()
    cfg = backbone(a.model) or {}
    label = a.label or cfg.get('label') or Path(a.model).name
    out = Path(a.out or HERE / 'out' / label)
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'run.log'
    arts = Path(a.artifacts or out / 'artifacts')
    methods = []
    for m in [x.strip() for x in a.methods.split(',') if x.strip()]:
        for name in (ALL if m == 'all' else [m]):
            if name not in methods:
                methods.append(name)
    unknown = [m for m in methods if m not in ALL + CONTROLS]
    if unknown:
        sys.exit(f'unknown methods: {unknown}; choose from {ALL + CONTROLS} or all')
    if 'original' not in methods and not a.no_original:
        methods = ['original'] + methods
    spae = a.spae_flags or cfg.get('spae')
    if spae is None and set(methods) & {'spae', 'random', 'oracle'}:
        sys.exit('no settings entry for this backbone: pass --spae-flags')
    think = ['--no-think'] if a.no_think or cfg.get('think', True) else []
    dm = ['--device-map', a.device_map] if a.device_map else []
    needed = {'caa': [arts / 'caa' / 'vectors.pt', arts / 'caa' / 'selection.jsonl'],
              'juice': [arts / 'juice' / a.variant / 'heads_v2.jsonl', arts / 'juice' / a.variant / 'selection_v2.jsonl'],
              'autopasta': [arts / 'autopasta' / a.variant / 'selection.jsonl', arts / 'autopasta' / 'spans_test.jsonl']}
    for m in methods:
        missing = [str(f) for f in needed.get(m, []) if not f.exists()]
        if missing:
            sys.exit(f'{m}: missing {missing}; fit it first: python fit.py --model {a.model} --methods {m}')
    limit = ['--limit', str(a.limit)] if a.limit else []
    rows_arg = ['--rows', str(Path(a.rows).resolve())] if a.rows else []
    py, D = a.python, str(a.data)
    seeds = a.seeds or [None]
    sampling = cfg.get('sampling', (0.7, 0.8, 20))

    def sample_flags(seed):
        if seed is None:
            return []
        return ['--sample-answer', '--seed', str(seed), '--temperature', str(sampling[0]), '--top-p', str(sampling[1]), '--top-k', str(sampling[2])]

    def path(method, seed):
        return out / (f'{method}.jsonl' if seed is None else f'sampled/{method}_seed{seed}.jsonl')

    def complete(file, need):
        return file.exists() and sum(1 for l in file.read_text().splitlines() if l.strip()) >= need

    rows = [json.loads(l) for l in Path(D).read_text().splitlines() if l.strip()]
    if a.rows:
        keep = {x.strip() for x in Path(a.rows).read_text().splitlines() if x.strip()}
        rows = [r for r in rows if r['row_id'] in keep]
    n_all, n_scored = len(rows), sum(r['direction'] != 'control' for r in rows)
    need = {'all': min(n_all, a.limit) if a.limit else n_all, 'scored': min(n_scored, a.limit) if a.limit else n_scored}
    call1 = out / 'call1.jsonl'
    if set(methods) & TWO_CALL and not complete(call1, need['all']):
        run([py, 'spae_call1.py', '--model', a.model, '--conditions', D, '--prompt-file', 'prompts/call1.txt', *think, *dm, *limit, *rows_arg, '--out', str(call1)], log)
    for seed in seeds:
        for m in methods:
            target = path(m, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            if complete(target, need['all' if m == 'original' and seed is None else 'scored']):
                print(f'skip {target} (complete)', flush=True)
                continue
            sf = sample_flags(seed)
            if m in TWO_CALL:
                arm = {'original': ['--arm', 'baseline'], 'spae': ['--arm', 'dual'], 'random': ['--arm', 'random'],
                       'oracle': ['--arm', 'dual', '--oracle', 'value', '--oracle-stance']}[m]
                # the greedy Original runs on every row (its neutral rows define the gate); a sampled Original on the scored rows only
                cells = [] if m == 'original' and seed is None else ['--cells', SCORED]
                run([py, 'spae_two_call.py', 'call2', '--model', a.model, '--conditions', D, '--call1', str(call1), '--prompt-file', 'prompts/call1.txt',
                     *arm, *cells, *SPAE_COMMON.split(), *spae.split(), *think, *dm, *limit, *rows_arg, *sf, '--out', str(target)], log)
            elif m in ('s2a', 'cad', 'adacad'):
                extra = ['--rewrites', str(path('s2a', None))] if m == 's2a' and seed is not None and path('s2a', None).exists() else []
                run([py, 'baselines/run_baseline.py', '--model', a.model, '--model-label', label, '--data', D, '--method', m, '--batch', str(a.batch),
                     *think, *dm, *limit, *rows_arg, *sf, *extra, '--out', str(target)], log)
            elif m == 'caa':
                run([py, 'baselines/caa.py', '--model', a.model, '--model-label', label, '--data', D, '--stage', 'test', '--vectors', str(arts / 'caa' / 'vectors.pt'),
                     '--selection', str(arts / 'caa' / 'selection.jsonl'), '--variant', a.variant, '--batch', str(a.batch), *think, *dm, *sf, '--out', str(target)]
                    + (['--rows', str(limit_rows(rows, a.limit, out))] if a.limit else rows_arg), log)
            elif m == 'juice':
                run([py, 'baselines/juice.py', '--stage', 'test', '--model', a.model, '--model-label', label, '--data', D, '--variant', a.variant,
                     '--artifacts', str(arts / 'juice' / a.variant), '--batch', str(a.batch), *think, *dm, *sf, '--out', str(target)]
                    + (['--rows', str(limit_rows(rows, a.limit, out))] if a.limit else rows_arg), log)
            elif m == 'autopasta':
                run([py, 'baselines/autopasta.py', '--stage', 'test', '--model', a.model, '--model-label', label, '--data', D, '--variant', a.variant,
                     '--artifacts', str(arts / 'autopasta' / a.variant), '--spans', str(arts / 'autopasta' / 'spans_test.jsonl'), '--batch', str(a.batch),
                     *think, *dm, *sf, '--out', str(target)] + (['--rows', str(limit_rows(rows, a.limit, out))] if a.limit else rows_arg), log)
    if a.skip_score:
        return
    names = {'original': 'Original', 'spae': 'SPAE', 'random': 'Random', 'oracle': 'Oracle', 's2a': 'S2A', 'cad': 'CAD', 'adacad': 'AdaCAD',
             'caa': 'CAA', 'juice': 'JuICE', 'autopasta': 'AutoPASTA'}
    arms = [f'{names[m]}={path(m, None)}' for m in methods if path(m, None).exists()]
    if not path('original', None).exists():
        sys.exit(f"no {path('original', None)}: run the original arm first, then score")
    if 'original' not in methods:
        arms = [f"Original={path('original', None)}"] + arms
    score = out / 'score.md'
    with open(score, 'w') as fh:
        subprocess.run([py, 'score.py', '--data', D, '--arms', *arms], stdout=fh, cwd=HERE, check=True)
    print(f'scores written to {score}', flush=True)
    if a.seeds:
        sampled = [f'{names[m]}={out}/sampled/{m}_seed{{seed}}.jsonl' for m in methods if all(path(m, s).exists() for s in a.seeds)]
        greedy = [f'{names[m]}={path(m, None)}' for m in methods if path(m, None).exists()]
        with open(out / 'sampled' / 'score.md', 'w') as fh:
            subprocess.run([py, 'score_sampled.py', '--data', D, '--gates', str(path('original', None)), '--seeds', *map(str, a.seeds),
                            '--arms', *sampled, '--greedy', *greedy], stdout=fh, cwd=HERE, check=True)
        print(f'sampled scores written to {out}/sampled/score.md', flush=True)


def limit_rows(rows, limit, out):
    """A row-id file with the first `limit` scored rows (both families interleaved), for the runners without --limit."""
    groups = [[r for r in rows if r['direction'] != 'control' and r['family'] == f] for f in ('F1', 'F3')]
    pick = [r['row_id'] for i in range(max(map(len, groups))) for g in groups for r in g[i:i + 1]][:limit]
    path = out / 'limit_rows.txt'
    path.write_text('\n'.join(pick) + '\n')
    return path


if __name__ == '__main__':
    main()
