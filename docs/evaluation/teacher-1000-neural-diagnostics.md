# Accepted pilot: neural diagnostic baseline

Measured 2026-10-21 00:28 UTC (October 20 evening in New York) using
**Inez Neural 0.1.0**, the accepted pilot artifact
`inez-29m-pilot/teacher/candidate-0001000`, with 29,228,779 parameters.
The complete machine-readable evidence is
[teacher-1000-baseline.json](teacher-1000-baseline.json).

Weights SHA-256:
`1af8a5a0267b484706a9e7fa4411450850bdfb96ddf60947e54542a0b0e2207e`.

| Diagnostic | Passed / total |
| --- | --- |
| Move choice, 32 simulations | 10 / 16 |
| Move choice, 128 simulations | 11 / 16 |
| Move choice, 512 simulations | 11 / 16 |
| Raw value comparisons | 0 / 6 |
| Terminal-rule controls | 8 / 8 |

The evaluation completed with exit code **2**, meaning behavioral failures were
measured successfully. All move searches returned legal moves and PVs with the
requested completed simulation count. Terminal controls are rules checks, not
evidence of learned tactical strength. The separate software regression run
passed all 11 CTest suites, including the new diagnostic runner tests.

## Observed weaknesses

- **Bishop safety:** in the position after 7...b5 from Xiyuan's neural game,
  Inez chooses O-O instead of Bb3 at every tested budget, for both colors.
- **Knight safety:** White chooses Nf7, leaving the knight capturable by the
  king, at all three budgets. The color-swapped case passes. The suite does not
  assume color symmetry in the learned network.
- **Underpromotion:** both colors promote to a queen and immediately stalemate
  at every budget. The tested root is explicitly limited to the four promotions.
- **Free rook:** the mirrored case misses the capture at 32 simulations but
  finds it at 128 and 512. Free queen captures pass throughout.
- **Value sensitivity:** all six comparisons fail their stated criteria. Some
  even rank the materially worse position above the better one. These failures
  concern the raw learned value, before search backup or UCI score conversion.

| Value pair | Better value | Worse value | Difference | Required difference |
| --- | ---: | ---: | ---: | ---: |
| Queen advantage, White | 0.3381 | 0.3209 | 0.0172 | 0.05 |
| Queen advantage, Black | -0.1832 | -0.1829 | -0.0003 | 0.05 |
| Rook advantage, White | 0.3134 | 0.3267 | -0.0133 | 0.05 |
| Rook advantage, Black | -0.1838 | -0.1827 | -0.0011 | 0.05 |
| Preserve bishop, White | -0.0397 | -0.0176 | -0.0221 | 0.05 |
| Preserve bishop, Black | -0.1261 | -0.1409 | 0.0148 | 0.05 |

Values are `P(win) - P(loss)` for the fixed player named in the row. Queen and
rook comparisons also require the advantage to be positive and the disadvantage
negative; none pass those combined checks. The bishop comparisons retain full
game history and invert side-to-move values correctly. The 0.05 margin is a
diagnostic choice, not a calibrated probability threshold.

The mate-in-one, specified poisoned-capture avoidance, and immediate stalemate
avoidance cases pass at all budgets. Passing a restricted avoidance check does
not certify general conversion skill.

## Provenance and interpretation

CPU, four PyTorch threads, Cpuct 2.5, Python 3.14.8, PyTorch 2.14.1,
python-chess 1.11.2, macOS arm64. Fixed simulation budgets make this a behavior
comparison; reported timings are incidental and include concurrent regression
testing. Stockfish 19 independently confirmed all 16 move labels and all six
value orderings using 100,000 nodes per position, one thread, 64 MiB hash and
full strength. Its executable checksum and returned scores are in the JSON.

The report's Git revision is the pre-change HEAD `c665252`; the suite and runner
were new working-tree files during measurement. Their SHA-256 checksums in the
report match the committed files. Model files and acceptance metadata were not
modified.

This small development suite includes the observed human-game mistake and
correlated mirrored cases. It provides **no Elo estimate**. The evidence supports
investigating training targets, value perspective and material/tactical coverage
before increasing model size. It does not by itself establish whether the cause
is data quality, training duration, architecture or another training defect.
Use separate unseen positions and paired engine matches to assess later gains.

See [the diagnostic guide](../neural-diagnostics.md) for rerunning and extending
the evaluation without altering the promotion gate.
