"""JuICE on CoPE-Bench: frozen head selection (`heads_v2.jsonl`), the 6 x 6 suppression / enhancement grid fitted on the
held-out development rows (`fit`, `select`) and the test run under the selected pair (`test`).

The development rows of a variant (fit-mix: all scored cells; fit-syco: wrong_claim; fit-stub: context_conflict) are
split into the profiling rows, whose ids are stored with the head selection, and the held-out rows used for fitting."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import add_model_args, add_sampling_args, full_attention_layers, hit, keep_rows, load, text_config   # noqa: E402
from juice_core import JuiceEngine                                                                               # noqa: E402
from run_baseline import finish_row                                                                              # noqa: E402

VARIANT_CELL = {'fit-mix': None, 'fit-syco': 'wrong_claim', 'fit-stub': 'context_conflict'}


def variant_rows(dev, variant):
    """The scored development rows a variant fits on."""
    cell = VARIANT_CELL[variant]
    return [r for r in dev.values() if r['direction'] != 'control' and (cell is None or r['control_type'] == cell)]


def partition(dev, variant, heads_path):
    """The profiling rows (in the stored order) and the held-out rows: the variant's development cells minus every row of the profiled questions."""
    ids = json.loads(Path(heads_path).read_text())['profile_row_ids']
    profile = [dev[i] for i in ids]
    items = {r['item_id'] for r in profile}
    held = [r for r in variant_rows(dev, variant) if r['item_id'] not in items]
    return profile, held


def main():
    """Select heads from a profile, fit the strength grid, select the pair, or run the test rows."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['heads', 'fit', 'select', 'test'], required=True)
    add_model_args(p)
    p.add_argument('--variant', choices=list(VARIANT_CELL), default='fit-mix')
    p.add_argument('--data', type=Path, required=True, help='test split')
    p.add_argument('--dev', type=Path, help='development split (fit / select / heads)')
    p.add_argument('--gates', nargs='+', help='select: Original run of the development split')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True, help='directory with heads_v2.jsonl, selection_v2.jsonl, profile/, fit_v2/')
    p.add_argument('--config-index', type=int)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--batch', type=int, default=1)
    add_sampling_args(p)
    a = p.parse_args()
    label = a.model_label or Path(a.model).name
    a.out.parent.mkdir(parents=True, exist_ok=True)
    heads_path = a.artifacts / 'heads_v2.jsonl'
    if a.stage != 'test':
        dev = load([a.dev])
        profile, held = partition(dev, a.variant, heads_path)
    if a.stage == 'heads':
        from transformers import AutoConfig
        cfg = text_config(AutoConfig.from_pretrained(a.model))
        full = full_attention_layers(cfg)
        rows = {}
        for path in sorted((a.artifacts / 'profile').glob('layers_*.jsonl')):
            manifest = json.loads(path.with_suffix('.config.json').read_text())
            assert manifest['profile_row_ids'] == [r['row_id'] for r in profile]
            for r in map(json.loads, path.read_text().splitlines()):
                key = (r['layer'], r['head'])
                assert key not in rows
                rows[key] = r
        assert set(rows) == {(l, h) for l in full for h in range(cfg.num_attention_heads)}, 'Incomplete profile'
        chosen = {}
        for sign in ['positive', 'negative']:
            pool = [r for r in rows.values() if r[sign + '_gain_sum'] > 0]
            pool.sort(key=lambda r: (-r[sign + '_gain_sum'], r['layer'], r['head']))
            # up to ten eligible heads; never filled with harmful ones
            chosen[sign + '_heads'] = [{'layer': r['layer'], 'head': r['head'], 'gain': r[sign + '_gain_sum']} for r in pool[:10]]
        result = {'row_id': 'heads', 'model': label, 'variant': a.variant, 'profile_row_ids': [r['row_id'] for r in profile],
                  'profile_item_ids': [r['item_id'] for r in profile],
                  'head_selection_rule': 'positive mean over the frozen alpha grid, rank by mean, up to ten; flat mixed dev objective',
                  'selected_counts': {k: len(v) for k, v in chosen.items()}, 'validation_rows': len(held), 'profiled_heads': len(rows),
                  'data_sha256': hashlib.sha256(a.dev.read_bytes()).hexdigest(), **chosen}
        a.out.write_text(json.dumps(result) + '\n')
        print('COMPLETE', flush=True)
        return
    heads = json.loads(heads_path.read_text())
    if a.stage == 'select':
        original = load(a.gates)
        controls = [r for r in dev.values() if r['direction'] == 'control']
        assert all(r['row_id'] in original for r in controls)
        eligible = {r['item_id'] for r in controls if hit(r, original[r['row_id']], 'gold')}
        scored = [r for r in held if r['item_id'] in eligible]
        assert scored
        records = []
        grid_idx = sorted(int(f.name[7:9]) for f in (a.artifacts / 'fit_v2').glob('config_*.jsonl'))
        assert grid_idx, 'no fit_v2 configs on disk'
        for idx in grid_idx:
            rows = load([a.artifacts / 'fit_v2' / f'config_{idx:02d}.jsonl'])
            assert set(rows) == {r['row_id'] for r in held}
            suppress, enhance = divmod(idx, 6)
            records.append({'config_index': idx, 'suppress': suppress, 'enhance': enhance,
                            'score': sum(hit(r, rows[r['row_id']], 'gold') for r in scored) / len(scored)})
        best = max(records, key=lambda x: (x['score'], -x['suppress'] - x['enhance'], -x['suppress'], -x['enhance']))
        a.out.write_text(json.dumps({'row_id': a.variant, 'model': label, 'variant': a.variant, 'scored_dev_rows': len(scored),
                                     'grid': records, 'tie_break': 'score, smaller total intervention, smaller suppression, smaller enhancement',
                                     **best}) + '\n')
        print('COMPLETE', flush=True)
        return
    if a.stage == 'fit':
        assert a.config_index is not None and 0 <= a.config_index < 36
        rows = held
        suppress, enhance = divmod(a.config_index, 6)
        datapath = a.dev
    else:
        selection = json.loads((a.artifacts / 'selection_v2.jsonl').read_text())
        suppress, enhance = selection['suppress'], selection['enhance']
        datapath = a.data
        rows = keep_rows([r for r in load([datapath]).values() if r['direction'] != 'control'], a.rows)
        i, n = map(int, a.shard.split('/'))
        assert 0 <= i < n
        rows = [r for j, r in enumerate(rows) if j % n == i]
    specs = [(x['layer'], x['head'], enhance) for x in heads['positive_heads']] + [(x['layer'], x['head'], -suppress) for x in heads['negative_heads']]
    signature = {'args': {k: str(v) for k, v in vars(a).items() if k != 'batch'},
                 'data_sha256': hashlib.sha256(datapath.read_bytes()).hexdigest(),
                 'heads_sha256': hashlib.sha256(heads_path.read_bytes()).hexdigest(),
                 'core_sha256': hashlib.sha256((HERE / 'juice_core.py').read_bytes()).hexdigest(),
                 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'suppress': suppress, 'enhance': enhance}
    cfg = a.out.with_suffix('.config.json')
    if cfg.exists():
        assert json.loads(cfg.read_text()) == signature, 'Incompatible resume'
    cfg.write_text(json.dumps(signature, indent=2) + '\n')
    done = load([a.out]) if a.out.exists() else {}
    e = JuiceEngine(a.model, a.cap_gib, a.no_think)
    e.configure_sampling(a)
    e.batch = a.batch
    prepared = [(r, e.encode(r['prompt'])) for r in rows if r['row_id'] not in done]
    prepared.sort(key=lambda x: (x[0]['instruction_style'], len(x[1]), x[0]['row_id']))
    with a.out.open('a', buffering=1) as fh:
        for style in ['short_phrase', 'cot_letter']:
            group = [r for r in prepared if r[0]['instruction_style'] == style]
            for offset in range(0, len(group), a.batch):
                chunk = group[offset:offset + a.batch]
                e.row_seed(chunk[0][0]['row_id'])
                tokens = e.juice([x[1] for x in chunk], 400 if style == 'cot_letter' else 32, specs)
                for (r, ids), gen in zip(chunk, tokens):
                    extra = {'model': label, 'method': 'juice-' + a.variant, 'suppress': suppress, 'enhance': enhance}
                    out = finish_row(e, r, gen, extra, lambda letters, ids=ids, gen=gen: e.juice_probe(ids, gen, letters, specs) if a.stage == 'test' else {})
                    if a.stage != 'test':
                        out['answer'] = out.get('stated_answer', out['answer'])
                    fh.write(json.dumps(out, ensure_ascii=False) + '\n')
                    done[r['row_id']] = out
                fh.flush()
                os.fsync(fh.fileno())
                print(json.dumps({'stage': a.stage, 'rows': len(done), 'expected': len(rows), 'variant': a.variant}), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
