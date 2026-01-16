#!/usr/bin/env python3
"""
Salience-guided TTA + per-example test-time adaptation for SemEval 2026 Task 4 (Track A).

This script is a faithful port of the logic in "untitled21 (1).py" with one IMPORTANT fix:
- In the exported notebook .py, diff_for_view() is decorated with @torch.no_grad(), which would
  *disable gradients* and make the per-example adaptation step a no-op / crash on backward.
  Here we keep no_grad for salience scoring and the *initial* diff aggregation, but we ENABLE
  gradients during the adaptation steps so LN+head can actually update.

Everything else (model arch, preprocessing, TTA view construction, adaptation objective, decision rule)
matches the notebook.

Output:
- Writes track_a.jsonl (default) in the same line order as input JSONL.
- Copies original fields and adds: "text_a_is_closer": true/false

Run:
  python "inference/inf_script.py" --model_path "best_model.pt" --input_jsonl "SemEval2026-Task_4-test-v1/test_track_a.jsonl" --output_jsonl track_a.jsonl --device cuda
"""

import argparse
import hashlib
import json
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# =========================
# Constants (match notebook)
# =========================
CHECKPOINT    = "microsoft/deberta-v3-large"
MAX_LEN       = 512
FREEZE_LAYERS = 4
POOLING       = "mean"   # "mean" or "cls"

ANC_WORDS  = 240
CAND_WORDS = 260

# Deterministic preprocessing applied in BOTH train & eval (label-free)
P_ENTITY_ANON = 0.60
P_DROP_PARENS = 0.35
STYLE_NORMALIZE_ALWAYS = True

# TTA + per-example adaptation
USE_SALIENCE_TTA = True
SALIENCE_TOPK_LIST   = (2, 4)
SALIENCE_KEEP_LAST   = 2
SALIENCE_KEEP_FIRST  = True
SALIENCE_MAX_SENTS   = 40
TTA_TRUNC_VIEWS      = ("headtail", "tail")
TTA_AGG              = "mean"   # mean over diffs

USE_PER_EX_ADAPT = True
ADAPT_STEPS = 3
ADAPT_LR = 1e-3
ADAPT_ONLY_IF_ABS_DIFF_LT = 0.20
ADAPT_VAR_W = 1.0
ADAPT_PSEUDO_W = 1.0
ADAPT_DRIFT_W = 0.10


# =========================
# spaCy loading (match notebook behavior)
# =========================
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

NLP = None


# =========================
# Utils (match notebook)
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    t = str(text).replace("\\", " ")
    return " ".join(t.split())

def hash_int(s: str) -> int:
    import hashlib
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def deterministic_gate(seed_int: int, p: float) -> bool:
    if p <= 0:
        return False
    if p >= 1:
        return True
    x = (seed_int % 10_000) / 10_000.0
    return x < p

def text_seed(text: str) -> int:
    return hash_int(clean_text(text))

def squash_runs(s: str, ch: str) -> str:
    out = []
    run = 0
    for c in s:
        if c == ch:
            run += 1
            if run <= 1:
                out.append(c)
        else:
            run = 0
            out.append(c)
    return "".join(out)

def remove_parentheticals(text: str) -> str:
    out = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return clean_text("".join(out))

def style_normalize(text: str, seed_int: int) -> str:
    t = clean_text(text)
    if not t:
        return ""
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = squash_runs(t, "!")
    t = squash_runs(t, "?")
    t = squash_runs(t, ".")
    if deterministic_gate(seed_int + 911, P_DROP_PARENS):
        t = remove_parentheticals(t)
    return clean_text(t)

def anonymize_entities_spacy(text: str) -> str:
    global NLP
    t = clean_text(text)
    if not t:
        return ""
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(t)
    allowed = {"PERSON", "GPE", "LOC", "ORG", "DATE", "TIME"}
    ents = [e for e in doc.ents if e.label_ in allowed]
    if not ents:
        return t

    mapping = {}
    counters = {lab: 0 for lab in allowed}
    pieces = []
    last = 0
    for e in ents:
        key = (e.text, e.label_)
        if key not in mapping:
            counters[e.label_] += 1
            mapping[key] = f"{e.label_}_{counters[e.label_]}"
        pieces.append(t[last:e.start_char])
        pieces.append(mapping[key])
        last = e.end_char
    pieces.append(t[last:])
    return clean_text("".join(pieces))

def preprocess_text(text: str) -> str:
    t = clean_text(text)
    if not t:
        return ""
    s = text_seed(t)

    if STYLE_NORMALIZE_ALWAYS:
        t = style_normalize(t, s)

    if deterministic_gate(s + 1337, P_ENTITY_ANON):
        t = anonymize_entities_spacy(t)

    return clean_text(t)

def truncate_view(text: str, max_words: int, mode: str, seed_int: int) -> str:
    t = clean_text(text)
    if not t:
        return ""
    words = t.split()
    if len(words) <= max_words:
        return t

    if mode == "head":
        return " ".join(words[:max_words])
    if mode == "tail":
        return " ".join(words[-max_words:])
    if mode == "rand":
        rng = random.Random(seed_int)
        start_max = max(0, len(words) - max_words)
        start = rng.randint(0, start_max) if start_max > 0 else 0
        return " ".join(words[start:start + max_words])

    half = max_words // 2
    return " ".join(words[:half]) + " ... " + " ".join(words[-half:])

def split_sentences(text: str) -> List[str]:
    global NLP
    t = clean_text(text)
    if not t:
        return []
    if NLP is None:
        NLP = load_spacy()
    doc = NLP(t)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents if sents else [t]


# =========================
# Model (match notebook)
# =========================
class PairwiseRanker(torch.nn.Module):
    def __init__(self, model_name: str, freeze_layers: int = 0, pooling: str = "mean"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        encoder = getattr(self.backbone, "encoder", None)
        if hasattr(self.backbone, "deberta"):
            encoder = self.backbone.deberta.encoder

        if encoder and freeze_layers > 0:
            for i, layer in enumerate(encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.norm = torch.nn.LayerNorm(hidden)
        self.scorer = torch.nn.Linear(hidden, 1)

    def forward(self, **tok):
        out = self.backbone(**tok)
        if self.pooling == "cls":
            rep = out.last_hidden_state[:, 0]
        else:
            mask = tok["attention_mask"].unsqueeze(-1).float()
            rep = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.scorer(self.norm(rep)).squeeze(-1)

def load_state_dict_safely(model: torch.nn.Module, path: str, device: str):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


# =========================
# Scoring helpers (match notebook; grad optional)
# =========================
def score_pairs(model: torch.nn.Module, tokenizer, device: str,
                anchors: List[str], cands: List[str], *, grad: bool) -> torch.Tensor:
    tok = tokenizer(anchors, cands, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    if grad:
        return model(**tok)
    with torch.no_grad():
        return model(**tok)


# =========================
# Salience-guided sentence TTA (match notebook)
# =========================
@torch.no_grad()
def sentence_salience_view(model: PairwiseRanker, tokenizer, device: str,
                           anchor: str, cand: str, topk: int,
                           keep_last: int = 2, keep_first: bool = True,
                           max_sents: int = 40) -> str:
    anchor_p = preprocess_text(anchor)
    cand_p = preprocess_text(cand)

    sents = split_sentences(cand_p)
    if not sents:
        return cand_p
    if len(sents) > max_sents:
        head = sents[: max_sents // 2]
        tail = sents[-(max_sents - len(head)):]
        sents = head + tail

    first_sent = sents[0:1]
    last_sents = sents[-keep_last:] if keep_last > 0 and len(sents) > keep_last else sents[-1:]
    mid_sents = sents[1:len(sents) - len(last_sents)] if len(sents) > (1 + len(last_sents)) else []

    if mid_sents:
        anc_rep = [anchor_p] * len(mid_sents)
        scores = score_pairs(model, tokenizer, device, anc_rep, mid_sents, grad=False).detach().cpu().numpy()
        top_idx = np.argsort(scores)[::-1][: min(topk, len(mid_sents))]
        chosen = [mid_sents[i] for i in sorted(top_idx)]
    else:
        chosen = []

    out = []
    if keep_first and first_sent:
        out.extend(first_sent)
    out.extend(chosen)
    out.extend(last_sents)

    seen = set()
    final = []
    for s in out:
        ss = clean_text(s)
        if ss and ss not in seen:
            seen.add(ss)
            final.append(ss)

    return clean_text(" ".join(final))


def build_tta_views_for_triplet(model: PairwiseRanker, tokenizer, device: str,
                               anc: str, a: str, b: str) -> List[Tuple[str, str, str]]:
    anc_p = preprocess_text(anc)
    anc_view = truncate_view(anc_p, ANC_WORDS, "headtail", text_seed(anc_p))

    views: List[Tuple[str, str, str]] = []

    for v in TTA_TRUNC_VIEWS:
        a_p = preprocess_text(a)
        b_p = preprocess_text(b)
        a_v = truncate_view(a_p, CAND_WORDS, v, text_seed(a_p))
        b_v = truncate_view(b_p, CAND_WORDS, v, text_seed(b_p))
        views.append((anc_view, a_v, b_v))

    if USE_SALIENCE_TTA:
        for k in SALIENCE_TOPK_LIST:
            a_sv = sentence_salience_view(
                model=model, tokenizer=tokenizer, device=device,
                anchor=anc_view, cand=a, topk=k,
                keep_last=SALIENCE_KEEP_LAST, keep_first=SALIENCE_KEEP_FIRST,
                max_sents=SALIENCE_MAX_SENTS,
            )
            b_sv = sentence_salience_view(
                model=model, tokenizer=tokenizer, device=device,
                anchor=anc_view, cand=b, topk=k,
                keep_last=SALIENCE_KEEP_LAST, keep_first=SALIENCE_KEEP_FIRST,
                max_sents=SALIENCE_MAX_SENTS,
            )
            a_sv = truncate_view(a_sv, CAND_WORDS, "headtail", text_seed(a_sv))
            b_sv = truncate_view(b_sv, CAND_WORDS, "headtail", text_seed(b_sv))
            views.append((anc_view, a_sv, b_sv))

        def last_k_sent_view(x: str, k: int = 4) -> str:
            xp = preprocess_text(x)
            sents = split_sentences(xp)
            tail = sents[-k:] if len(sents) >= k else sents
            return clean_text(" ".join(tail)) if tail else xp

        a_tail = truncate_view(last_k_sent_view(a, 4), CAND_WORDS, "headtail", text_seed(a))
        b_tail = truncate_view(last_k_sent_view(b, 4), CAND_WORDS, "headtail", text_seed(b))
        views.append((anc_view, a_tail, b_tail))

    dedup = []
    seen = set()
    for av, aa, bb in views:
        key = hashlib.md5((av + "||" + aa + "||" + bb).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            dedup.append((av, aa, bb))
    return dedup


# =========================
# Per-example adaptation (match notebook objective; FIX gradients)
# =========================
def get_adapt_params(model: PairwiseRanker):
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("norm.") or n.startswith("scorer."):
            params.append(p)
    return params

def snapshot_adapt_params(model: PairwiseRanker):
    snap = {}
    for n, p in model.named_parameters():
        if n.startswith("norm.") or n.startswith("scorer."):
            snap[n] = p.detach().clone()
    return snap

def restore_adapt_params(model: PairwiseRanker, snap: dict):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])

def diffs_for_views(model: PairwiseRanker, tokenizer, device: str,
                    views: List[Tuple[str, str, str]], *, grad: bool) -> torch.Tensor:
    anc = [v[0] for v in views]
    a   = [v[1] for v in views]
    b   = [v[2] for v in views]
    s_a = score_pairs(model, tokenizer, device, anc, a, grad=grad)
    s_b = score_pairs(model, tokenizer, device, anc, b, grad=grad)
    return s_a - s_b

def predict_with_tta_and_per_example_adapt(model: PairwiseRanker, tokenizer, device: str,
                                          anc: str, a: str, b: str) -> float:
    model.eval()
    views = build_tta_views_for_triplet(model, tokenizer, device, anc, a, b)

    # initial diffs across views (no adaptation, no grads)
    with torch.no_grad():
        diffs0 = diffs_for_views(model, tokenizer, device, views, grad=False).detach()
        mean0 = diffs0.mean()
        agg0 = mean0.item()

    if (not USE_PER_EX_ADAPT) or (abs(agg0) >= ADAPT_ONLY_IF_ABS_DIFF_LT) or (len(views) < 2):
        return agg0

    y = 1.0 if agg0 > 0 else 0.0
    y_sign = 1.0 if y == 1.0 else -1.0

    snap = snapshot_adapt_params(model)
    params = get_adapt_params(model)
    opt = torch.optim.Adam(params, lr=ADAPT_LR)

    for n, p in model.named_parameters():
        if n.startswith("norm.") or n.startswith("scorer."):
            p.requires_grad = True
        else:
            p.requires_grad = False

    try:
        for _ in range(ADAPT_STEPS):
            model.eval()

            diffs = diffs_for_views(model, tokenizer, device, views, grad=True)
            mean = diffs.mean()
            var = diffs.var(unbiased=False)

            pseudo_loss = F.softplus(-y_sign * diffs).mean()
            drift = (mean - mean0).pow(2)

            loss = ADAPT_VAR_W * var + ADAPT_PSEUDO_W * pseudo_loss + ADAPT_DRIFT_W * drift

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            diffs1 = diffs_for_views(model, tokenizer, device, views, grad=False).detach()
            agg = diffs1.mean().item() if TTA_AGG == "mean" else diffs1.max().item()

    finally:
        restore_adapt_params(model, snap)
        for _, p in model.named_parameters():
            p.requires_grad = True

    return agg


# =========================
# I/O
# =========================
def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "anchor_text" not in obj or "text_a" not in obj or "text_b" not in obj:
                continue
            rows.append(obj)
    return rows

def write_jsonl(rows: List[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", default="track_a.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)

    model = PairwiseRanker(CHECKPOINT, freeze_layers=FREEZE_LAYERS, pooling=POOLING).to(device)
    load_state_dict_safely(model, args.model_path, device)
    model.eval()

    rows = load_jsonl(args.input_jsonl)
    if not rows:
        raise RuntimeError(f"No valid rows read from {args.input_jsonl}")

    out_rows = []
    for i, obj in enumerate(rows):
        diff = predict_with_tta_and_per_example_adapt(
            model=model, tokenizer=tokenizer, device=device,
            anc=obj["anchor_text"], a=obj["text_a"], b=obj["text_b"]
        )
        new_obj = dict(obj)
        new_obj["text_a_is_closer"] = bool(diff > 0)
        out_rows.append(new_obj)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(rows)}")

    write_jsonl(out_rows, args.output_jsonl)
    print(f"Wrote {len(out_rows)} predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
