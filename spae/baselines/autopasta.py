"""AutoPASTA on CoPE-Bench: key-sentence extraction by the model (`extract`), mapping to a token span (`map`), the
coarse-to-fine head search on the development split (`coarse`, `rank`, `fine`, `candidates`, `fit`, `select`) with a
prefill-only read-out, and the test run under the selected heads (`test`).

The search stages and the test stage read the spans from a mapping file (`--spans`, the output of `extract` + `map` on
the same split); `fit.py` runs every stage in order for one backbone."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import add_model_args, add_sampling_args, full_attention_layers, hit, keep_rows, load, options, text_config   # noqa: E402
from readout import ANSWER_SUFFIX, letter_token_ids                                                                        # noqa: E402
from run_baseline import finish_row                                                                                        # noqa: E402

METRIC = 'mean gold probability: F1 fixed ANSWER_SUFFIX letter; F3 first gold token; Original-gated dev; prefill only'
VARIANT_CELL = {'fit-mix': None, 'fit-syco': 'wrong_claim', 'fit-stub': 'context_conflict'}


def ranked(records):
    """Records by descending probability, ties by layer then head."""
    return sorted(records, key=lambda r: (-r['probability'], r['layer'], r.get('head', -1)))


def make_candidates(coarse, fine):
    """The 28 head configurations: per-layer top-k and pooled top-k over the best 3 to 6 layers."""
    layers = [r['layer'] for r in ranked(coarse)[:6]]
    assert len(layers) == 6
    out = []
    for n in (3, 4, 5, 6):
        active = layers[:n]
        pool = [r for r in fine if r['layer'] in active]
        for count in (4, 6, 8):
            heads = {l: [r['head'] for r in ranked([r for r in pool if r['layer'] == l])[:count]] for l in active}
            out.append({'config_index': len(out), 'kind': 'per_layer', 'layers': n, 'k': count, 'heads': heads})
        for count in (16, 24, 32, 64):
            heads = {}
            for r in ranked(pool)[:count]:
                heads.setdefault(r['layer'], []).append(r['head'])
            out.append({'config_index': len(out), 'kind': 'pooled', 'layers': n, 'k': count, 'heads': heads})
    assert len(out) == 28
    return out


def main():
    """Run one AutoPASTA stage."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['extract', 'map', 'coarse', 'rank', 'fine', 'candidates', 'fit', 'select', 'test'], required=True)
    add_model_args(p)
    p.add_argument('--variant', choices=list(VARIANT_CELL), default='fit-mix')
    p.add_argument('--data', type=Path, required=True, help='the split the stage runs on (test split for test; development split otherwise)')
    p.add_argument('--gates', nargs='+', help='search stages: Original run of the development split')
    p.add_argument('--spans', type=Path, help='mapping file (output of --stage map) for the search and test stages')
    p.add_argument('--extracted', type=Path, help='map: the output of --stage extract')
    p.add_argument('--artifacts', type=Path, help='directory with rank.jsonl, candidates.jsonl, selection.jsonl and the coarse/, fine/, fit/ outputs')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--layer-start', type=int, default=0, help='coarse: first layer (default 0)')
    p.add_argument('--layer-end', type=int, help='coarse: one past the last layer (default: the layer count)')
    p.add_argument('--rank-index', type=int, help='fine: one of the six ranked layers, written to --out (default: all six, written to --out/rank_N.jsonl)')
    p.add_argument('--config-index', type=int, help='fit: one of the 28 candidates, written to --out (default: every candidate, written to --out/config_XX.jsonl)')
    p.add_argument('--configs', help='fit: comma list of candidate indices instead of all 28')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int)
    p.add_argument('--batch', type=int, default=1)
    add_sampling_args(p)
    a = p.parse_args()
    label = a.model_label or Path(a.model).name
    art = a.artifacts
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def write_single(r):
        a.out.write_text(json.dumps(r) + '\n')
        print('COMPLETE', flush=True)

    data = {r['row_id']: r for r in keep_rows(list(load([a.data]).values()), a.rows)}
    rows = [r for r in data.values() if r['direction'] != 'control']
    bare_map = {r['item_id']: r['prompt'] for r in data.values() if r['direction'] == 'control'}
    if a.stage in ('extract', 'map', 'test'):
        rows = keep_rows(rows, a.rows)
        i, n = map(int, a.shard.split('/'))
        assert 0 <= i < n
        rows = [r for j, r in enumerate(rows) if j % n == i]
    if a.limit:
        rows = rows[:a.limit]
    if a.stage == 'map':
        from autopasta_core import PastaEngine
        e = PastaEngine(a.model, a.cap_gib, a.no_think, a.device_map)
        extracted = load([a.extracted])
        done = load([a.out]) if a.out.exists() else {}
        with a.out.open('a', buffering=1) as fh:
            for r in rows:
                if r['row_id'] in done:
                    continue
                rec = e.map_sentence(extracted[r['row_id']])
                rec.update(model=label, prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest())
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print('COMPLETE', flush=True)
        return
    if a.stage == 'extract':
        from autopasta_core import PastaEngine
        e = PastaEngine(a.model, a.cap_gib, a.no_think, a.device_map)
        e.batch = a.batch
        done = load([a.out]) if a.out.exists() else {}
        todo = [r for r in rows if r['row_id'] not in done]
        with a.out.open('a', buffering=1) as fh:
            for start in range(0, len(todo), a.batch):
                for rec in e.extract(todo[start:start + a.batch], bare_map):
                    rec['model'] = label
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fh.flush()
                print(json.dumps({'stage': 'extract', 'rows': min(start + a.batch, len(todo)), 'expected': len(todo)}), flush=True)
        print('COMPLETE', flush=True)
        return
    from transformers import AutoConfig
    modelcfg = text_config(AutoConfig.from_pretrained(a.model))
    nl, nh, full = modelcfg.num_hidden_layers, modelcfg.num_attention_heads, full_attention_layers(modelcfg)
    if a.stage == 'rank':
        recs = list(load(sorted((art / 'coarse').glob('layers_*.jsonl'))).values())
        assert {r['layer'] for r in recs} == set(full)
        write_single({'row_id': 'rank', 'metric': METRIC, 'layers': [r['layer'] for r in ranked(recs)[:6]], 'coarse': recs})
        return
    if a.stage == 'candidates':
        coarse = json.loads((art / 'rank.jsonl').read_text())
        fine = list(load(sorted((art / 'fine').glob('rank_*.jsonl'))).values())
        assert {(r['layer'], r['head']) for r in fine} == {(l, h) for l in coarse['layers'] for h in range(nh)}
        write_single({'row_id': 'candidates', 'metric': METRIC, 'configs': make_candidates(coarse['coarse'], fine)})
        return
    if a.stage == 'select':
        configs = json.loads((art / 'candidates.jsonl').read_text())['configs']
        fits = [json.loads(f.read_text()) for f in sorted((art / 'fit').glob('config_*.jsonl'))]
        assert fits and {r['config_index'] for r in fits} <= set(range(28))
        best = max(fits, key=lambda r: (r['probability'], -sum(map(len, configs[r['config_index']]['heads'].values())), -r['config_index']))
        write_single({'row_id': 'selection', 'metric': METRIC, 'variant': a.variant, 'grid': fits, 'selected': configs[best['config_index']],
                      'dev_prefill_probability': best['probability'], 'dev_prefill_target_rate': best['target_rate'],
                      'selected_head_count': sum(map(len, configs[best['config_index']]['heads'].values()))})
        return
    if a.stage != 'test':
        assert all(r['split'] == 'dev' for r in rows), 'the search stages run on the development split'
        cell = VARIANT_CELL[a.variant]
        if cell:
            rows = [r for r in rows if r['control_type'] == cell]
        original = load(a.gates)
        controls = [r for r in data.values() if r['direction'] == 'control']
        assert all(r['row_id'] in original for r in controls)
        eligible = {r['item_id'] for r in controls if hit(r, original[r['row_id']], 'gold')}
        rows = [r for r in rows if r['item_id'] in eligible]
        assert rows
    maps = load([a.spans])
    assert all(r['row_id'] in maps and maps[r['row_id']]['prompt_sha256'] == hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows), 'mapping file does not cover these rows'
    deps = {'data': hashlib.sha256(a.data.read_bytes()).hexdigest(), 'mappings': hashlib.sha256(a.spans.read_bytes()).hexdigest()}
    for filename in ['rank.jsonl', 'candidates.jsonl', 'selection.jsonl']:
        if art and (art / filename).exists():
            deps[filename] = hashlib.sha256((art / filename).read_bytes()).hexdigest()
    used = {'fine': ['rank.jsonl'], 'fit': ['candidates.jsonl'], 'test': ['selection.jsonl']}.get(a.stage, [])
    deps = {k: v for k, v in deps.items() if k in ['data', 'mappings'] + used}
    sig = {'args': {k: str(v) for k, v in vars(a).items() if k not in ('batch', 'rank_index', 'config_index', 'configs')}, 'deps': deps, 'metric': METRIC,
           'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'core': hashlib.sha256((HERE / 'autopasta_core.py').read_bytes()).hexdigest()}
    cfg = a.out.with_suffix('.config.json')
    if cfg.exists():
        assert json.loads(cfg.read_text()) == sig, 'Incompatible resume'
    cfg.write_text(json.dumps(sig, indent=2) + '\n')
    from autopasta_core import PastaEngine
    e = PastaEngine(a.model, a.cap_gib, a.no_think, a.device_map)
    e.configure_sampling(a)
    e.batch = a.batch
    prepared = [(r, e.encode(r['prompt']), maps[r['row_id']]['highlight_tokens']) for r in rows]
    prepared.sort(key=lambda x: (x[0]['instruction_style'], len(x[1]), x[0]['row_id']))

    @torch.inference_mode()
    def evaluate(heads):
        """Prefill-only read-out of one head configuration: mean gold probability and target rate over the gated rows."""
        probs, correct = [], []
        for r, ids, span in prepared:
            gold = r['answers']['gold']
            gold = gold[0] if isinstance(gold, list) else gold
            if r['instruction_style'] == 'cot_letter':
                letters = letter_token_ids(e.tok, ANSWER_SUFFIX, options(r['prompt']))
                seq, target, allowed = ids + e.tok.encode(ANSWER_SUFFIX, add_special_tokens=False), letters[gold], list(letters.values())
            else:
                seq, target, allowed = ids, e.tok.convert_tokens_to_ids(e.tok.tokenize(gold)[0]), None
            # the fixed read-out suffix is part of the prefill input, as in PASTA's input boundary
            with e.steering([seq], [span], heads):
                t = torch.tensor([seq], device=e.device)
                logits = e.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            probs.append(float(logits.softmax(-1)[0, target]))
            correct.append(int((allowed[int(logits[0, allowed].argmax())] if allowed else int(logits[0].argmax())) == target))
        return {'probability': sum(probs) / len(probs), 'target_rate': sum(correct) / len(correct), 'rows': len(probs), 'metric': METRIC}

    def run_into(path, jobs):
        """Evaluate the (row_id, record function) jobs not yet in `path`, appending each record (resumable)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        done = load([path]) if path.exists() else {}
        with path.open('a', buffering=1) as fh:
            for rid, fn in jobs:
                if rid in done:
                    continue
                r = fn()
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
                print(json.dumps({'stage': a.stage, 'file': str(path), 'row_id': rid}), flush=True)

    if a.stage == 'coarse':
        if a.layer_end is None:
            a.layer_end = nl
        assert 0 <= a.layer_start < a.layer_end <= nl
        run_into(a.out, [(str(l), lambda l=l: {'row_id': str(l), 'layer': l, **evaluate({l: list(range(nh))})}) for l in range(a.layer_start, a.layer_end) if l in full])
    elif a.stage == 'fine':
        ranked_layers = json.loads((art / 'rank.jsonl').read_text())['layers']
        for ri in ([a.rank_index] if a.rank_index is not None else range(len(ranked_layers))):
            l = ranked_layers[ri]
            run_into(a.out if a.rank_index is not None else a.out / f'rank_{ri}.jsonl',
                     [(f'{l}:{h}', lambda l=l, h=h: {'row_id': f'{l}:{h}', 'layer': l, 'head': h, **evaluate({l: [h]})}) for h in range(nh)])
    elif a.stage == 'fit':
        configs = json.loads((art / 'candidates.jsonl').read_text())['configs']
        indices = [a.config_index] if a.config_index is not None else [int(x) for x in a.configs.split(',')] if a.configs else list(range(len(configs)))
        for ci in indices:
            run_into(a.out if a.config_index is not None else a.out / f'config_{ci:02d}.jsonl',
                     [(str(ci), lambda ci=ci: {'row_id': str(ci), 'config_index': ci, **evaluate(configs[ci]['heads'])})])
    else:
        heads = json.loads((art / 'selection.jsonl').read_text())['selected']['heads']
        done = load([a.out]) if a.out.exists() else {}
        with a.out.open('a', buffering=1) as fh:
            for style in ['short_phrase', 'cot_letter']:
                group = [x for x in prepared if x[0]['instruction_style'] == style and x[0]['row_id'] not in done]
                for start in range(0, len(group), a.batch):
                    chunk = group[start:start + a.batch]
                    e.row_seed(chunk[0][0]['row_id'])
                    gens = e.pasta([x[1] for x in chunk], 400 if style == 'cot_letter' else 32, [x[2] for x in chunk], heads)
                    for (r, ids, span), gen in zip(chunk, gens):
                        extra = {'model': label, 'method': 'autopasta-' + a.variant, 'selected_head_count': sum(map(len, heads.values()))}
                        out = finish_row(e, r, gen, extra, lambda letters, ids=ids, gen=gen, span=span: e.pasta_probe(ids, gen, letters, span, heads))
                        out['aligned'] = bool(span)
                        fh.write(json.dumps(out, ensure_ascii=False) + '\n')
                        done[r['row_id']] = out
                    fh.flush()
                    os.fsync(fh.fileno())
                    print(json.dumps({'stage': 'test', 'rows': len(done), 'expected': len(prepared)}), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
