# Plan: break the char-tiny plateau

**Status at session-end (2026-05-09):** reign 1, king-loss=3.41 (ppl 30), 4
evals in history. Last 3 trained challengers all *lost* despite their
training-batch-loss being lower (~2.95) than the king's eval-loss (3.41).

## Diagnosis

Each `--train --train-steps 500 --train-batch-size 8` run samples 500 ×
8 = 4 000 sequences with replacement from the 541-sequence shard. Per
sequence:

```
P(seen at least once) = 1 - (540/541)^4000 ≈ 0.9994     (per token-position)
P(seen at least once over WHOLE 2048-window) — closer to ~99% per
sequence after 500 batches of 8
```

But the *training loss reported* is the **last batch's loss only** (single
sample). With 124k-param model + 541 windows, the model memorizes its
recent batch (low loss) without generalizing to the full eval distribution.

Root causes (most-likely first):

1. **Catastrophic forgetting from previous king's weights.** Constant LR
   3e-4 on top of an already-converged checkpoint causes oscillation —
   the optimizer "shakes off" earlier-king's learned features.
2. **Random sampling with replacement** + reporting only last-step loss
   gives a misleading picture of how well the model *generalizes* to the
   eval distribution.
3. **No regularization.** AdamW with `weight_decay=0` allows memorization.
4. **Eval ≈ train distribution but with different sampling.** Validator's
   bootstrap test samples a different N=100 indices than the random ones
   the miner happened to gradient-update.

## Acceptance criteria for "broke the plateau"

- King-loss in dashboard falls below **3.0** (ppl < 20)
- At least **3 successful dethrones** in the history after this plan
- Validation: hold-out test (Phase G) shows train/eval gap < 0.3 nats
  (i.e. the model genuinely learned, didn't just memorize)

## Phased approach

Each phase is independent. Run in order; if Phase N succeeds, skip to
"compounding runs" (Phase Z). If it fails, log to snapshot, then move
to Phase N+1.

Snapshot before each phase via:
```
python -m playground.experiments.tracker snapshot --name char-tiny
```

### Phase A: longer training (10x steps)

**Hypothesis:** with 5000 steps, every window is seen ~70 times — full
coverage, real learning.

**Commands** (after the standard char-tiny bring-up — see
project_nano_gpt_devnet memory):
```
python -m playground.launch_miner --wallet miner_beta --push --train \
    --train-steps 5000 --train-batch-size 8 \
    --shakespeare-npy playground/dataset/data-char/shards/shard_000000.npy
```

**Expected runtime:** ~5 min per submission (vs ~50s for 500 steps).

**Decision criteria:** if next reveal dethrones with king-loss < 3.3,
**proceed to Phase Z** (compounding). If it loses again, snapshot and
go to Phase B.

### Phase B: larger batch + lower LR

**Hypothesis:** smaller per-step gradient variance + smaller updates
preserve the previous king's features.

```
python -m playground.launch_miner --wallet miner_beta --push --train \
    --train-steps 2000 --train-batch-size 32 --train-lr 1e-4 \
    --shakespeare-npy playground/dataset/data-char/shards/shard_000000.npy
```

batch=32 × seq_len=2048 × vocab=69 = 4.4M element logits = 17 MB fp32.
Easily fits in Pascal's 6 GB. Pascal can compute it in ~150ms/step →
~5 min for 2000 steps.

**Decision criteria:** dethrone with king-loss < 3.2 → Phase Z. Else
Phase C.

### Phase C: cosine LR schedule + weight decay

Adds standard fine-tuning regularization. Requires modifying
`launch_miner.train_on_shakespeare` to accept LR scheduler + AdamW
weight_decay (currently it only accepts a single scalar `lr`). ~30 min
of `playground/`-only code (no skarp kode change).

Pseudocode:
```python
optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
warmup = LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
cosine = CosineAnnealingLR(optimizer, T_max=n_steps - warmup_steps,
                            eta_min=peak_lr * 0.01)
scheduler = SequentialLR(optimizer, [warmup, cosine],
                          milestones=[warmup_steps])
```

CLI surface to add:
- `--train-lr-warmup 100`
- `--train-lr-min 1e-5`
- `--train-weight-decay 0.01`

**Decision criteria:** dethrone with king-loss < 3.0 → Phase Z. Else
Phase D.

### Phase D: deterministic full-pass training

Replace `randint`-sampling with a shuffle-then-iterate loop. Guarantees
each sequence is seen exactly N/541 times across N steps, no
under-coverage tail.

```python
def train_epochs(model, tokens, *, n_epochs, seq_len, batch_size, lr):
    n_seq = len(tokens) // seq_len
    starts = np.arange(0, n_seq * seq_len, seq_len)
    for epoch in range(n_epochs):
        np.random.shuffle(starts)
        for batch_starts in batched(starts, batch_size):
            batch = make_batch(tokens, batch_starts, seq_len)
            train_step(model, batch, optimizer)
```

CLI: `--train-epochs 20` (replaces `--train-steps`).

**Decision criteria:** dethrone with king-loss < 3.0 → Phase Z. Else
Phase E.

### Phase E: validation split + early stopping

Add `--val-fraction 0.1`. Train on first 90% of windows, monitor loss
on the held-out 10% every N steps. Stop when val-loss stops improving
for 3 consecutive checks.

This won't make the model BETTER — but it tells us *whether* the
plateau is from overfitting or from the model's true capacity ceiling.
If train/val gap widens fast, we're memorizing → smaller model needed.
If val plateau matches train plateau, we've hit Chinchilla floor for
this model.

```
python -m playground.launch_miner --wallet miner_beta --push --train \
    --train-steps 5000 --train-batch-size 32 --train-lr 1e-4 \
    --val-fraction 0.1 \
    --shakespeare-npy playground/dataset/data-char/shards/shard_000000.npy
```

This phase is **diagnostic** — log the train/val gap, then decide if F
or G is the right next move.

### Phase F: bigger model (loosen Chinchilla)

If Phase E shows train/val gap is small (< 0.2 nats), the model is at
its capacity ceiling. Bump n_layer 4 → 8, n_embd 32 → 64. ~480k params,
1.9 MB. Re-seed king as `ai-garage/Teutonic-Nano-Char-medium-king`,
add new experiment `char-medium`.

This is a NEW experiment, not a continuation. Use the experiment-
creation pattern from `playground/experiments/char-tiny/` as template.

### Phase G: more data (real Chinchilla)

If Phase E shows train/val gap is large (> 0.5 nats), the model is
memorizing. More data is the fix. Build
`playground/dataset/build_culturax.py` mirroring `build_shakespeare.py`
but pulling ~10-100M tokens from `uonlp/CulturaX`. Re-tokenize with
the existing char tokenizer (or a 4k-vocab BPE if preferring word-piece).

This is the path toward "real language model" — big enough corpus that
the 124k-param model is forced to compress patterns rather than
memorize sequences. Expected to push king-loss toward 1.5-2.0 nats
(ppl 4.5-7).

### Phase Z: compounding runs (after any successful dethrone)

Once a phase produces a successful dethrone with low loss, run 5-10
more submissions with the SAME config alternating between miner_beta
and miner_alpha to build out a clean falling curve in the dashboard.

```
for i in 1 2 3 4 5; do
  miner=$( [ $((i % 2)) -eq 1 ] && echo miner_beta || echo miner_alpha )
  python -m playground.launch_miner --wallet $miner --push --train \
    --train-steps <best> --train-batch-size <best> --train-lr <best> \
    --shakespeare-npy playground/dataset/data-char/shards/shard_000000.npy
  sleep 30
done
python -m playground.experiments.tracker snapshot --name char-tiny
```

## Stop conditions

- **Hard time budget:** 2 hours total. If no phase has produced a
  sub-3.0 king-loss by then, write up findings + move to a different
  experiment (likely Phase F or G).
- **GPU thermals:** Pascal P3200 throttles at ~85°C. If `nvidia-smi`
  shows sustained > 80°C, reduce batch or insert sleeps between runs.

## Memory hooks

When this plan is run in a clean session, the tracker should emit a
phase-result snapshot. Update
`~/.claude/projects/-home-roy-mining-sn03/memory/project_nano_gpt_devnet.md`
with the empirically-best (steps, batch, lr, weight-decay) combo —
that becomes the default for char-tiny going forward, and informs
Phase 1 of future Chinchilla-targeted experiments.
