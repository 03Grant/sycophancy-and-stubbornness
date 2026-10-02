"""Fit the three fitted baselines (CAA, JuICE, AutoPASTA) for one backbone on the development split, from scratch.

    python fit.py --model Qwen/Qwen3.5-9B                      # every fitted baseline, variant fit-mix
    python fit.py --model Qwen/Qwen3.5-9B --methods caa,juice   # a subset
    python fit.py --model Qwen/Qwen3.5-9B --device-map auto     # a backbone spread over several GPUs

Stages, in order: the Original run of the development split (its correctly answered neutral rows are the fitting gate);
CAA (vectors from the official pairs, downloaded on first use; the layer x multiplier scan; the selection); JuICE (four
profiling rows drawn from the gated rows, the per-head profile, the head lists, the 36-pair grid on the held-out rows, the
selection); AutoPASTA (key-sentence extraction and mapping on the development and the test split, the coarse / fine
head search, the 28 candidates, their fit, the selection). Everything lands in out/<label>/artifacts/, where run.py
expects it; every stage is resumable, so an interrupted fit continues where it stopped.
"""
import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'baselines'))
from common import hit, keep_rows, load   # noqa: E402
from run import BACKBONES, backbone       # noqa: E402

FITTED = ['caa', 'juice', 'autopasta']
VARIANT_CELL = {'fit-mix': None, 'fit-syco': 'wrong_claim', 'fit-stub': 'context_conflict'}


def run(cmd, log):
    """Run one stage, teeing its output to a log file; abort on failure."""
    print('+', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    with open(log, 'a') as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    if proc.returncode:
        sys.exit(f'stage failed (see {log})')


def complete(path, need):
    """Whether a jsonl output already holds `need` rows."""
    return path.exists() and sum(1 for l in path.read_text().splitlines() if l.strip()) >= need


def profile_rows(rows, eligible, variant, seed=42, per_family=2):
    """The JuICE profiling rows: per task family, `per_family` gated rows of the variant's cells, drawn at random with
    distinct conditions and distinct questions; when the family has too few gated questions to leave one for the
    held-out fit, the rows come from one question (a small smoke-test split)."""
    rng = random.Random(seed)
    cell = VARIANT_CELL[variant]
    chosen = []
    for family in ('F1', 'F3'):
        pool = [r for r in rows if r['family'] == family and r['direction'] != 'control' and r['item_id'] in eligible
                and (cell is None or r['control_type'] == cell)]
        rng.shuffle(pool)
        distinct_items = len({r['item_id'] for r in pool}) > per_family
        picked = []
        for r in pool:
            if len(picked) == per_family:
                break
            if cell is None and any(r['control_type'] == c['control_type'] for c in picked):
                continue
            if any((r['item_id'] == c['item_id']) == distinct_items for c in picked):
                continue
            picked.append(r)
        chosen += picked
    return chosen


def main():
    """Run the fitting stages of the requested baselines in dependency order."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Hugging Face id or local path of the backbone')
    p.add_argument('--methods', default='all', help="comma list of caa, juice, autopasta, or 'all'")
    p.add_argument('--dev', default=str(HERE.parent / 'CoPE-Bench' / 'cope_bench_dev.jsonl'))
    p.add_argument('--test', default=str(HERE.parent / 'CoPE-Bench' / 'cope_bench_test.jsonl'), help='test split (AutoPASTA maps the spans of its rows here)')
    p.add_argument('--out', help='output directory (default out/<label>; the artifacts go to its artifacts/ subdirectory)')
    p.add_argument('--label', help='backbone label (default: from the settings table, else the model name)')
    p.add_argument('--variant', default='fit-mix', choices=list(VARIANT_CELL))
    p.add_argument('--no-think', action='store_true', help='force the thinking switch off (default: from the settings table)')
    p.add_argument('--device-map', help='spread the model over several GPUs with accelerate, e.g. auto')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--rows', help='row-id file: fit on these development rows only (whole questions, neutral rows included); a smoke test')
    p.add_argument('--test-rows', help='row-id file for the test split (AutoPASTA span mapping)')
    p.add_argument('--caa-layers', default='all', help="CAA scan: 'all', 'middle', a range 'a-b' or a comma list")
    p.add_argument('--caa-multipliers', default='-2,-1.5,-1,-0.5,0.5,1,1.5,2')
    p.add_argument('--juice-alphas', default='1,3,5,10,30,-1,-2,-5')
    p.add_argument('--juice-layers', help="JuICE profile: a range 'a-b' (one past the last layer) instead of every layer; a smoke test")
    p.add_argument('--juice-configs', help='JuICE: comma list of the 36 pair indices to fit instead of all')
    p.add_argument('--pasta-configs', help='AutoPASTA: comma list of the 28 candidate indices to fit instead of all')
    p.add_argument('--skip-test-spans', action='store_true', help='AutoPASTA: do not extract and map the test split')
    p.add_argument('--python', default=sys.executable)
    a = p.parse_args()
    cfg = backbone(a.model) or {}
    label = a.label or cfg.get('label') or Path(a.model).name
    out = Path(a.out or HERE / 'out' / label)
    art = out / 'artifacts'
    art.mkdir(parents=True, exist_ok=True)
    log = out / 'fit.log'
    methods = []
    for m in [x.strip() for x in a.methods.split(',') if x.strip()]:
        for name in (FITTED if m == 'all' else [m]):
            if name not in methods:
                methods.append(name)
    unknown = [m for m in methods if m not in FITTED]
    if unknown:
        sys.exit(f'unknown methods: {unknown}; choose from {FITTED} or all')
    think = ['--no-think'] if a.no_think or cfg.get('think', True) else []
    dm = ['--device-map', a.device_map] if a.device_map else []
    rows_arg = ['--rows', str(Path(a.rows).resolve())] if a.rows else []
    py, DEV, TEST = a.python, str(Path(a.dev).resolve()), str(Path(a.test).resolve())
    common = ['--model', a.model, '--model-label', label, *think, *dm, '--batch', str(a.batch)]
    dev_rows = keep_rows([json.loads(l) for l in Path(DEV).read_text().splitlines() if l.strip()], a.rows)

    # 1. the Original run of the development split: the fitting gate
    gate = art / 'dev_original.jsonl'
    if not complete(gate, len(dev_rows)):
        run([py, 'baselines/run_baseline.py', *common, '--data', DEV, '--method', 'original', *rows_arg, '--out', str(gate)], log)
    original = load([gate])
    eligible = {r['item_id'] for r in dev_rows if r['direction'] == 'control' and hit(r, original[r['row_id']], 'gold')}
    print(json.dumps({'stage': 'gate', 'eligible_questions': len(eligible), 'dev_rows': len(dev_rows)}), flush=True)
    if not eligible:
        sys.exit('no eligible development questions: the Original run answers no neutral row correctly')

    # 2. CAA: vectors, scan, selection
    if 'caa' in methods:
        sel = art / 'caa' / 'selection.jsonl'
        if not sel.exists():
            run([py, 'baselines/caa.py', *common, '--data', DEV, '--stage', 'fit', '--vectors', str(art / 'caa' / 'vectors.pt'), '--gates', str(gate),
                 '--layers', a.caa_layers, '--multipliers', a.caa_multipliers, *rows_arg, '--out', str(sel)], log)
        print(json.dumps({'stage': 'caa', 'selection': json.loads(sel.read_text().splitlines()[0])['layer']}), flush=True)

    # 3. JuICE: profiling rows, profile, heads, grid, selection
    if 'juice' in methods:
        jart = art / 'juice' / a.variant
        jart.mkdir(parents=True, exist_ok=True)
        prof = jart / 'profile_rows.txt'
        if not prof.exists():
            picked = profile_rows(dev_rows, eligible, a.variant)
            if not picked:
                sys.exit('no gated development rows to profile JuICE on')
            prof.write_text(''.join(r['row_id'] + '\n' for r in picked))
        jcommon = [*common, '--data', TEST, '--dev', DEV, '--variant', a.variant, '--artifacts', str(jart), '--profile-rows', str(prof), *rows_arg]
        layers = ['--layer-start', a.juice_layers.split('-')[0], '--layer-end', a.juice_layers.split('-')[1]] if a.juice_layers else []
        name = 'layers_' + (a.juice_layers.replace('-', '_') if a.juice_layers else 'all')
        heads = jart / 'heads_v2.jsonl'
        if not heads.exists():
            run([py, 'baselines/juice.py', '--stage', 'profile', *jcommon, '--alphas', a.juice_alphas, *layers, '--out', str(jart / 'profile' / f'{name}.jsonl')], log)
            run([py, 'baselines/juice.py', '--stage', 'heads', *jcommon, '--out', str(heads)], log)
        sel = jart / 'selection_v2.jsonl'
        if not sel.exists():
            configs = ['--configs', a.juice_configs] if a.juice_configs else []
            run([py, 'baselines/juice.py', '--stage', 'fit', *jcommon, *configs, '--out', str(jart / 'fit_v2')], log)
            run([py, 'baselines/juice.py', '--stage', 'select', *jcommon, '--gates', str(gate), '--out', str(sel)], log)
        print(json.dumps({'stage': 'juice', **{k: json.loads(sel.read_text())[k] for k in ('suppress', 'enhance', 'score')}}), flush=True)

    # 4. AutoPASTA: spans of both splits, coarse / fine search, candidates, fit, selection
    if 'autopasta' in methods:
        part = art / 'autopasta'
        pvar = part / a.variant
        pvar.mkdir(parents=True, exist_ok=True)
        splits = [('dev', DEV, rows_arg, dev_rows)]
        if not a.skip_test_spans:
            test_rows = keep_rows([json.loads(l) for l in Path(TEST).read_text().splitlines() if l.strip()], a.test_rows)
            splits.append(('test', TEST, ['--rows', str(Path(a.test_rows).resolve())] if a.test_rows else [], test_rows))
        for split, data, rarg, rows in splits:
            spans = part / f'spans_{split}.jsonl'
            need = sum(r['direction'] != 'control' for r in rows)
            if not complete(spans, need):
                extracted = part / f'extracted_{split}.jsonl'
                if not complete(extracted, need):
                    run([py, 'baselines/autopasta.py', '--stage', 'extract', *common, '--data', data, *rarg, '--out', str(extracted)], log)
                run([py, 'baselines/autopasta.py', '--stage', 'map', *common, '--data', data, '--extracted', str(extracted), *rarg, '--out', str(spans)], log)
        pcommon = [*common, '--data', DEV, '--variant', a.variant, '--gates', str(gate), '--spans', str(part / 'spans_dev.jsonl'), '--artifacts', str(pvar), *rows_arg]
        sel = pvar / 'selection.jsonl'
        if not sel.exists():
            if not (pvar / 'rank.jsonl').exists():
                run([py, 'baselines/autopasta.py', '--stage', 'coarse', *pcommon, '--out', str(pvar / 'coarse' / 'layers_all.jsonl')], log)
                run([py, 'baselines/autopasta.py', '--stage', 'rank', *pcommon, '--out', str(pvar / 'rank.jsonl')], log)
            if not (pvar / 'candidates.jsonl').exists():
                run([py, 'baselines/autopasta.py', '--stage', 'fine', *pcommon, '--out', str(pvar / 'fine')], log)
                run([py, 'baselines/autopasta.py', '--stage', 'candidates', *pcommon, '--out', str(pvar / 'candidates.jsonl')], log)
            configs = ['--configs', a.pasta_configs] if a.pasta_configs else []
            run([py, 'baselines/autopasta.py', '--stage', 'fit', *pcommon, *configs, '--out', str(pvar / 'fit')], log)
            run([py, 'baselines/autopasta.py', '--stage', 'select', *pcommon, '--out', str(sel)], log)
        print(json.dumps({'stage': 'autopasta', 'selected_head_count': json.loads(sel.read_text())['selected_head_count']}), flush=True)
    print(f'artifacts in {art}', flush=True)


if __name__ == '__main__':
    main()
