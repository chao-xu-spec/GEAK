#!/usr/bin/env python3
"""Decode step time by batch size, read from the serving logs of an A/B.

A throughput comparison collapses the whole run into one number, and its noise band is a
policy threshold. The serving engine's own log carries a finer record: every few dozen
decode iterations sglang prints the running batch size, the mean accepted length and the
generation throughput, from which the time of one decode step follows:

    step_ms = running_requests * accept_len / gen_throughput * 1000

Grouped by running batch size and compared between the stock and candidate legs, this shows
whether the candidate made the common decode steps slower (or faster) even when the whole
run lands inside the noise band. The median per batch size is used, which discards the
steps that shared their log window with a prefill.

    decode_step.py STOCK_LOG[,STOCK_LOG...] CANDIDATE_LOG[,CANDIDATE_LOG...] [--min-samples N]

Only sglang's "Decode batch" lines are understood (with or without speculative decoding;
only rank TP0's lines are read); other engines yield no rows and the comparison is skipped.
"""
import argparse
import json
import re
import statistics
import sys

# sglang prints the decode line in several dialects (rank prefixes such as `ATTN_CP0 TP0`,
# `Decode batch.` or `Decode batch [n],`, `#token` or `#full token`, an `accept len` only under
# speculative decoding), so each field is looked up on its own instead of matching one layout.
RUNNING = re.compile(r"Decode batch[^,]*[,.]\s*#running-req: (\d+)")
ACCEPT = re.compile(r"accept len: ([\d.]+)")
THROUGHPUT = re.compile(r"gen throughput \(token/s\): ([\d.]+)")
RANK = re.compile(r"\bTP(\d+)\b")
MIN_THROUGHPUT = 5.0        # lines logged while the server was idle carry no step time
MIN_SAMPLES = 30


def step_times(path, min_throughput=MIN_THROUGHPUT):
    """{running batch size: [step_ms, ...]} from rank 0's decode lines of one log."""
    steps = {}
    with open(path, errors="replace") as f:
        for line in f:
            if "Decode batch" not in line:
                continue
            rank = RANK.search(line)
            if rank and int(rank.group(1)) != 0:
                continue
            m, rate = RUNNING.search(line), THROUGHPUT.search(line)
            if not m or not rate:
                continue
            accepted = ACCEPT.search(line)
            # without speculative decoding every running request advances one token per step
            running, accept, throughput = int(m.group(1)), float(accepted.group(1)) if accepted else 1.0, float(rate.group(1))
            if throughput < min_throughput or running < 1 or accept <= 0:
                continue
            steps.setdefault(running, []).append(running * accept / throughput * 1000.0)
    return steps


def pooled(paths):
    out = {}
    for path in paths:
        for bs, values in step_times(path).items():
            out.setdefault(bs, []).extend(values)
    return out


def compare(stock_paths, candidate_paths, min_samples=MIN_SAMPLES):
    """One row per batch size that both sides observed at least `min_samples` times."""
    stock, candidate = pooled(stock_paths), pooled(candidate_paths)
    rows = []
    for bs in sorted(set(stock) & set(candidate)):
        if len(stock[bs]) < min_samples or len(candidate[bs]) < min_samples:
            continue
        s, c = statistics.median(stock[bs]), statistics.median(candidate[bs])
        rows.append({"batch": bs, "stock_ms": round(s, 3), "candidate_ms": round(c, 3),
                     "ratio": round(c / s, 4), "n_stock": len(stock[bs]),
                     "n_candidate": len(candidate[bs])})
    return rows


def regressions(rows, threshold_pct):
    """The rows whose candidate step is slower than stock by more than `threshold_pct`."""
    return [r for r in rows if r["ratio"] > 1.0 + threshold_pct / 100.0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stock", help="comma-separated server logs of the stock legs")
    ap.add_argument("candidate", help="comma-separated server logs of the candidate legs")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    a = ap.parse_args(argv)
    rows = compare(a.stock.split(","), a.candidate.split(","), a.min_samples)
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
