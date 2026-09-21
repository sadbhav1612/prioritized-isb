#!/usr/bin/env python3
"""The two tests used in the paper, on per-seed values.

  paired  exact two-sided paired sign-flip test. Enumerates all 2^n sign
          assignments of the per-seed differences, so it is exact for any n;
          its smallest attainable p is 2 / 2^n (0.002 at n = 10, 0.0625 at 5).
  record  permutation test for a best-of-seeds statistic: pool the per-seed
          values of all methods, draw groups of the method's size at random,
          and report how often a group's minimum is at least as low as the
          method's. This is how the paper judges a fastest lap.

  python scripts/signflip.py paired --a 5.40 5.31 ... --b 5.33 5.29 ...
  python scripts/signflip.py record --method 5.1 4.9 ... --pool 5.3 5.0 ...
"""
import argparse
import itertools

import numpy as np


def paired_signflip(a, b):
    """Two-sided exact p for mean(a - b) = 0, pairing a[i] with b[i]."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n, m = len(d), abs(d.mean())
    hits = sum(1 for s in itertools.product((1.0, -1.0), repeat=n)
               if abs(np.dot(d, s) / n) >= m - 1e-12)
    return hits / 2.0 ** n


def record_permutation(method, pool, draws=200000, seed=0):
    """Fraction of random groups of len(method) drawn from `pool` whose
    minimum is <= min(method). `pool` should include the method's values."""
    method = np.asarray(method, float)
    pool = np.asarray(pool, float)
    rng = np.random.default_rng(seed)
    k, best = len(method), method.min()
    mins = np.array([rng.choice(pool, size=k, replace=False).min()
                     for _ in range(draws)])
    return float((mins <= best + 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paired")
    p.add_argument("--a", type=float, nargs="+", required=True)
    p.add_argument("--b", type=float, nargs="+", required=True)
    r = sub.add_parser("record")
    r.add_argument("--method", type=float, nargs="+", required=True)
    r.add_argument("--pool", type=float, nargs="+", required=True)
    r.add_argument("--draws", type=int, default=200000)
    args = ap.parse_args()
    if args.cmd == "paired":
        if len(args.a) != len(args.b):
            ap.error("--a and --b need one value per seed, in the same order")
        print("mean difference %+.4f, p = %.4f (n = %d)"
              % (np.mean(np.subtract(args.a, args.b)),
                 paired_signflip(args.a, args.b), len(args.a)))
    else:
        print("P(random group minimum <= method minimum) = %.3f"
              % record_permutation(args.method, args.pool, args.draws))


if __name__ == "__main__":
    main()
