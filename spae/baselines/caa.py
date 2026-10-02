"""CAA: a per-layer mean-difference vector from the official 1,000 sycophancy pairs, added to the residual stream at one
layer with one multiplier; the (layer, multiplier) pair is selected on the development split (`--stage fit`, every
layer by default, `--layers middle` for the five middle layers, or a range / list) and then held fixed (`--stage test`)."""
import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

PAIRS_URL = 'https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/sycophancy/generate_dataset.json'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import add_model_args, add_sampling_args, decoder_layers, hit, keep_rows, load, model_text_config   # noqa: E402
from run_baseline import Engine, finish_row                                                                       # noqa: E402


def answer_tokens(tok, question, answer):
    """Chat-template ids of one pair and the position of the answer letter inside them."""
    text = tok.apply_chat_template([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}], tokenize=False)
    start = text.rfind(answer.strip())
    if start < 0:
        raise ValueError('Missing assistant answer')
    offset = start + next(i for i, c in enumerate(answer.strip()) if c in 'AB')
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    index = next(i for i, (a, b) in enumerate(enc['offset_mapping']) if a <= offset < b)
    return enc['input_ids'], index


@torch.inference_mode()
def vectors(engine, pairs_path, out, batch):
    """The per-layer vectors: loaded from `out` when it exists (checked against the pairs file when one is given), else
    computed as the mean residual difference at the answer letter, sycophantic minus truthful."""
    if out.exists():
        old = torch.load(out, map_location='cpu', weights_only=True)
        if pairs_path is not None and old['data_sha256'] != hashlib.sha256(pairs_path.read_bytes()).hexdigest():
            raise ValueError('CAA pair checksum changed')
        return old['vectors']
    if pairs_path is None:
        pairs_path = out.parent / 'official_caa_sycophancy_1000.json'
        if not pairs_path.exists():
            import urllib.request
            print(f'downloading the official CAA sycophancy pairs to {pairs_path}', flush=True)
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PAIRS_URL, pairs_path)
    pairs = json.loads(pairs_path.read_text())
    assert len(pairs) == 1000
    fingerprint = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    sums = torch.zeros(len(decoder_layers(engine.model)), model_text_config(engine.model).hidden_size, dtype=torch.float64)
    state, handles = {}, []
    for li, block in enumerate(decoder_layers(engine.model)):
        def capture(_module, _inputs, output, li=li):
            h = output[0] if isinstance(output, tuple) else output
            values = h[torch.arange(h.shape[0], device=h.device), state['positions'].to(h.device)]
            sums[li].add_(state['sign'] * values.detach().double().sum(0).cpu())
        handles.append(block.register_forward_hook(capture))
    try:
        for start in range(0, len(pairs), batch):
            for key, sign in [('answer_matching_behavior', 1), ('answer_not_matching_behavior', -1)]:
                encoded = [answer_tokens(engine.tok, p['question'], p[key]) for p in pairs[start:start + batch]]
                seqs = [x[0] for x in encoded]
                ids, mask = engine.pad(seqs)
                positions = mask.long().cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 0)
                state.update(sign=sign, positions=torch.tensor([ids.shape[1] - len(seq) + pair[1] for seq, pair in zip(seqs, encoded)], device=ids.device))
                engine.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=1)
            if start % 40 == 0:
                print(json.dumps({'stage': 'caa_vectors', 'pairs': min(start + batch, 1000)}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    result = (sums / 1000).float()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    torch.save({'vectors': result, 'data_sha256': fingerprint, 'orientation': 'sycophancy_minus_truthful',
                'scale': 'raw_mean_difference', 'n_pairs': 1000}, tmp)
    tmp.replace(out)
    return result


def scan_layers(spec, n_layers):
    """The layer indices a fit scans: every layer for 'all', the five middle layers for 'middle', else an inclusive range 'a-b' or a comma-separated list."""
    if spec == 'middle':
        center = n_layers // 2
        return list(range(center - 2, center + 3))
    if spec == 'all':
        return list(range(n_layers))
    if '-' in spec:
        a, b = map(int, spec.split('-'))
        layers = list(range(a, b + 1))
    else:
        layers = [int(x) for x in spec.split(',') if x.strip()]
    if not layers or any(not 0 <= l < n_layers for l in layers):
        raise ValueError(f'Layers {spec} outside 0..{n_layers - 1}')
    return layers


@contextmanager
def steer(engine, vector, layer, multiplier, prompt_width):
    """Add multiplier * vector to the residual stream of `layer` from the final prompt token onwards."""
    value = vector.to(device=engine.device, dtype=engine.model.dtype) * multiplier

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # prefill: apply from the final prompt token onwards; cached generation: each new token
        start = prompt_width - 1 if hidden.shape[1] >= prompt_width else 0
        changed = hidden.clone()
        changed[:, start:, :] += value.to(hidden.device)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = decoder_layers(engine.model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def evaluate(engine, rows, vec, layer, multiplier, path, batch, label, probe=True):
    """Generate every row under one (layer, multiplier) setting, appending to `path`; returns all rows on disk."""
    done = load([path]) if path.exists() else {}
    prepared = [(r, engine.encode(r['prompt'])) for r in rows if r['row_id'] not in done]
    prepared.sort(key=lambda x: (x[0]['instruction_style'], len(x[1]), x[0]['row_id']))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', buffering=1) as fh:
        for style in ('short_phrase', 'cot_letter'):
            group = [x for x in prepared if x[0]['instruction_style'] == style]
            for a in range(0, len(group), batch):
                chunk = group[a:a + batch]
                seqs = [x[1] for x in chunk]
                engine.row_seed(chunk[0][0]['row_id'])
                with steer(engine, vec, layer, multiplier, max(map(len, seqs))):
                    gens = engine.ordinary(seqs, 400 if style == 'cot_letter' else 32)
                for (r, ids), tokens in zip(chunk, gens):
                    def probe_fn(letters, ids=ids, tokens=tokens):
                        if not probe:
                            return {}
                        with steer(engine, vec, layer, multiplier, len(ids)):
                            return engine.probe(ids, tokens, letters)
                    out = finish_row(engine, r, tokens, {'model': label, 'method': 'caa', 'layer': layer, 'multiplier': multiplier}, probe_fn)
                    if not probe:
                        out['answer'] = out.get('stated_answer', out['answer'])
                    fh.write(json.dumps(out, ensure_ascii=False) + '\n')
                    done[r['row_id']] = out
                fh.flush()
                os.fsync(fh.fileno())
                print(json.dumps({'stage': 'caa_eval', 'file': str(path), 'rows': len(done)}), flush=True)
    return done


def main():
    """Fit the (layer, multiplier) pair on the development split, or run the selected pair on the test rows."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['fit', 'test'], required=True)
    p.add_argument('--pairs', type=Path, help='the official 1,000 sycophancy pairs (default: downloaded next to --vectors when the vectors have to be built; checked against an existing vectors file)')
    p.add_argument('--vectors', type=Path, required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--gates', nargs='+', help='fit: Original run of the development split (its neutral rows define the fitting gate)')
    p.add_argument('--selection', type=Path, help='test: the selection file written by --stage fit')
    p.add_argument('--variant', choices=['fit-mix', 'fit-syco', 'fit-stub'], default='fit-mix')
    p.add_argument('--layers', default='all', help="fit: layers to scan: 'all' (every layer, default), 'middle' (the five middle layers), a range 'a-b' (inclusive) or a comma list")
    p.add_argument('--multipliers', default='-2,-1.5,-1,-0.5,0.5,1,1.5,2', help='fit: comma list of multipliers')
    p.add_argument('--shard', default='0/1')
    add_sampling_args(p)
    a = p.parse_args()
    label = a.model_label or Path(a.model).name
    data = {r['row_id']: r for r in keep_rows(list(load([a.data]).values()), a.rows)}
    rows = [r for r in data.values() if r['direction'] != 'control']
    if a.stage == 'fit' and any(r['split'] != 'dev' for r in data.values()):
        raise ValueError('Fit requires the development split only')
    engine = Engine(a.model, a.cap_gib, a.no_think, a.device_map)
    engine.configure_sampling(a)
    engine.batch = a.batch
    vecs = vectors(engine, a.pairs, a.vectors, a.batch)
    if a.stage == 'test':
        chosen = load([a.selection])[a.variant]
        i, n = map(int, a.shard.split('/'))
        rows = [r for j, r in enumerate(rows) if j % n == i]
        evaluate(engine, rows, vecs[chosen['layer']], chosen['layer'], chosen['multiplier'], a.out, a.batch, label)
        print('COMPLETE', flush=True)
        return
    original = load(a.gates)
    controls = [r for r in data.values() if r['direction'] == 'control']
    if any(r['row_id'] not in original for r in controls):
        raise ValueError('Incomplete development gates')
    eligible = {r['item_id'] for r in controls if hit(r, original[r['row_id']], 'gold')}
    layers = scan_layers(a.layers, len(vecs))
    multipliers = [float(x) for x in a.multipliers.split(',')]
    all_scores = []
    for layer in layers:
        for multiplier in multipliers:
            path = a.out.parent / 'scan' / f'layer{layer}_mult{multiplier:g}.jsonl'
            print(json.dumps({'stage': 'caa_scan', 'layer': layer, 'multiplier': multiplier}), flush=True)
            predictions = evaluate(engine, rows, vecs[layer], layer, multiplier, path, a.batch, label, probe=False)
            scores = {}
            for variant, cell in [('fit-mix', None), ('fit-syco', 'wrong_claim'), ('fit-stub', 'context_conflict')]:
                selected = [r for r in rows if r['item_id'] in eligible and (cell is None or r['control_type'] == cell)]
                if not selected:
                    raise ValueError('No eligible fitting rows')
                scores[variant] = sum(hit(r, predictions[r['row_id']], 'gold') for r in selected) / len(selected)
            all_scores.append({'layer': layer, 'multiplier': multiplier, 'scores': scores})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in ('fit-mix', 'fit-syco', 'fit-stub'):
        chosen = max(all_scores, key=lambda x: (x['scores'][variant], -abs(x['multiplier']), -x['layer'], -x['multiplier']))
        records.append({'row_id': variant, 'model': label, 'layer': chosen['layer'], 'multiplier': chosen['multiplier'],
                        'dev_score': chosen['scores'][variant], 'gated_items': len(eligible), 'layers_scanned': layers, 'scan': all_scores,
                        'vector_scale': 'raw_mean_difference', 'tie_break': 'score, smaller_abs_multiplier, lower_layer, lower_multiplier'})
    tmp = a.out.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in records))
    tmp.replace(a.out)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
