"""Call 1 with a two-slot wording, plain greedy generation (no attention hook): the same chat-template scaffold and prefilled
"User pressure:" cue as spae_two_call.py call1, so the copied lines are identical; only the attention-derived masks are omitted.

    python spae_call1.py --model <path> --conditions ../CoPE-Bench/cope_bench_test.jsonl --prompt-file prompts/call1.txt --out call1.jsonl --no-think
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scaffold as K                                        # noqa: E402
from spae_two_call import CUE, EVIDENCE_TAG, parse_dual     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--no-think", action="store_true", help="pass enable_thinking=False to the chat template (thinking-mode models)")
    p.add_argument("--cells")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", help="i/n: keep every n-th record starting at i, so shards can run on separate GPUs")
    args = p.parse_args()
    instruction = args.prompt_file.read_text().strip()
    records = [json.loads(l) for l in args.conditions.read_text().splitlines() if l.strip()]
    if args.cells:
        keep = set(args.cells.split(","))
        records = [r for r in records if r.get("control_type") in keep]
    records = records[: args.limit] if args.limit else records
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        records = [r for j, r in enumerate(records) if j % n == i]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["row_id"] for l in args.out.read_text().splitlines() if l.strip()}
    records = [r for r in records if r["row_id"] not in done]
    print(f"{len(records)} rows to run ({len(done)} on disk)", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.no_think:
        K.CHAT_KWARGS = {"enable_thinking": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    trailing = K.trailing_turn_tokens(tok)
    t0 = time.time()
    with args.out.open("a") as fh, torch.no_grad():
        for i, rec in enumerate(records):
            ids, a, b = K.build_stage1(tok, rec["prompt"], instruction, CUE, trailing)
            inp = torch.tensor([ids], device=model.device)
            out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
            gen_ids = out[0, len(ids):].tolist()
            text = tok.decode(gen_ids, skip_special_tokens=True)
            # keep the generation up to the end of the Information line, as the watcher in spae_two_call.py does
            head, sep, tail = text.partition(EVIDENCE_TAG)
            if sep:
                tail_line = tail.lstrip("\n").split("\n")[0]
                text = head + sep + tail.lstrip("\n")[: len(tail_line)]
            stance, evidence = parse_dual(text)
            route = "BOTH" if stance and evidence else "MEMORY" if stance else "CONTEXT" if evidence else "NONE"
            row = {"row_id": rec["row_id"], "item_id": rec["item_id"], "source": rec.get("source"), "family": rec["family"],
                   "direction": rec.get("direction"), "control_type": rec.get("control_type"), "route": route, "router": "file",
                   "stance": stance, "evidence": evidence, "raw": text,
                   "stance_in_prompt": bool(stance) and K.quote_in_prompt(stance, rec["prompt"]),
                   "evidence_in_prompt": bool(evidence) and K.quote_in_prompt(evidence, rec["prompt"]),
                   "n_gen_tokens": len(gen_ids), "truncated": len(gen_ids) >= args.max_new_tokens and not sep,
                   # call 2 reads these: no attention masks (quote localisation only) and no label positions
                   "mask_m": [], "mask_e": [], "label_keys": {"stance": [], "evidence": [], "question": []}, "route_expected": None}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            if i % 20 == 0 or i == len(records) - 1:
                print(f"[{i + 1}/{len(records)}] {time.time() - t0:.0f}s {rec['row_id']} {rec['family']}:{rec.get('control_type')} route={route} "
                      f"stance={stance[:40]!r} evidence={evidence[:40]!r}", flush=True)


if __name__ == "__main__":
    main()
