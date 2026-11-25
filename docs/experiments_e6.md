# E6 — cost (2026-11-09)

E6 runs nothing. It reads the cost of E1 and E3 from what has already been
recorded (`scripts/report_e6.py` → `outputs/experiments/e6/results.json`).

## Sources

- **Calls and tokens:** they come from the `results/*.json` files. The
  `campaign_result` sums the usage that each LLM call writes to the ledger. In
  B4, the sum includes both models, attacker and Defender, without separation.
- **Time of one proposal:** it is the ledger's `elapsed_s`, which covers the
  generation of the traces for the search seeds, the measurement and the
  provider latency.
- **Retraining and cross matrix:** neither records its own duration. The time
  is estimated from the difference between the write times of consecutive
  files:
  - retraining: `D{r}.json` minus `round{r-1}.json`, only for r = 2 and 3;
  - campaign and test: `round{r}.json` minus `D{r}.json`;
  - matrix: difference between consecutive `rep*.json` files.

  The median is used, discarding intervals of 120 min or more: resumed runs
  leave gaps of hours, which are pauses, not work.

## Results

| Unit | n | Calls | Tokens | Time | Validity |
|---|---|---|---|---|---|
| E1 B1 (campaign) | 20 | 0 | 0 | 6.5 min (38.7 s/proposal) | 1.00 |
| E1 B2 (stops at 6 proposals) | 20 | 0 | 0 | 2.8 min | 1.00 |
| E1 B3 (120b) | 20 | 10 | 26.9k (24.2k–29.3k) | 10.1 min (60.5 s/proposal) | 0.86 |
| E1 B4 (120b + 20b) | 10 | 20.3 | 70.5k | 13.1 min (78.8 s/proposal) | 0.91 |
| E3 B1, round | 60 | 0 | 0 | retraining 2.9 + campaign and test 6.4 min | 1.00 |
| E3 B1 without defense, round | 60 | 0 | 0 | 0.3 + 2.0 min (traces cached) | 1.00 |
| E3 B3, round | 15 | 10 | 25.8k | 3.5 (n=7) + 7.0 min | 0.93 |
| B1 matrix (with placebo) | 20 | 0 | 0 | 30.5 min | — |
| B3 matrix | 5 | 0 | 0 | 13.2 min (median of only 3 intervals) | — |
| Defender plan, E3 (20b) | 28 | 1 | 5.0k | — | 27/28 accepted |

**Defender in B4:** there were 83 new queries, of which 80 were accepted; 7
plans were reused and 10 proposals had no valid candidate.

**Total tokens:** 1.77M LLM tokens.
- **120b:** about 1.21M, which add up from:
  - E1 B3 (538k);
  - E3 B3 (387k);
  - the attacker's share of B4 (705k − 83 × 5.0k).
- **20b:** about 0.56M. The B4 share is estimated from the ~5k per plan.

Of the B3 total, 81% is input.

**Compute:** about 35 h on one workstation, adding up:
- E1 campaigns: 8.7 h;
- E3-B1 lineages: 9.4 h with defense and 2.3 h without;
- matrices: 10.4 h;
- E3-LLM: 3.7 h.

The value matches the ~9.5 h observed in `campaign --arm B1`.

## Reading

1. **Generation dominates the time, not the LLM.** The LLM adds about 22 s per
   proposal, and the Defender another 18 s.
2. **The token cost is predictable**, because the context size is stable. One
   plan costs as much as two proposals, which is why B4 spends 2.6× the tokens
   of B3 with no measurable gain (B4−B3 −0.003).
3. **The bottleneck is Groq's daily quota.** The 120b needed about 6 days of
   quota, and the 20b about 3, against 35 h of compute.

## Caveats

- The retraining and matrix times are estimates from write times.
- The first retraining (D1) has no start mark.
- The trace cache makes repeated configurations cheaper, which is why the arm
  without defense comes out almost free.
- There is no monetary cost: everything ran on the free tier, and provider
  prices change.
