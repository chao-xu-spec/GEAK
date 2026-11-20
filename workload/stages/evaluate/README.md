# Evaluate

Compare the declared reference and candidate systems using their native serving receipts.
[Native E2E](native-e2e/SKILL.md) is the initial implementation; another implementation
must satisfy the same workload, quality and comparison contract.

Declare the decision this comparison supports. A scoped native screen may return matched
selection evidence under the [direct-call contract](../CONTRACT.md#direct-module-calls);
its restricted protocol and coverage are not full task acceptance. Complete comparisons
still use the finalizer below; a screen cannot replace required adoption evidence.

`finalize.py` owns the shared A/B calculation and receipt admission. It consumes the
existing `e2e/bench_summary.json` convention, including an admissible fresh `stock_control`
(the unchanged reference for this comparison), and the configured `native_validation`.
`reference_baseline`, when supplied in `PARAMS`, names the historical reference receipt;
otherwise the standard recipe's `RUN_ROOT/baseline/out/baseline.json` is used. Relative
configured paths resolve from the repository root.

The E2E decision threshold is `PARAMS.noise_band`, then the historical receipt's
`noise_band`, then `stages.baseline.params.noise_band` in `config/default.json`.
Explicit scenario values remain overrides. This is a policy threshold, separate from
kernel `keep_threshold` and measured repeat variability. When changing policy for an
existing comparison, retain its receipts and old verdict and re-evaluate in a new `OUT`.

Beside the verdict, `ab.json` and `REPORT.md` report what the comparison can and cannot show:
the spread measured between the legs of one arm (`measured_noise_pct`) with a warning when the
configured band is more than three times wider, which would report a repeatable change as
`no_change`; for an applyback, each installed kernel's share of GPU time (from
`roofline/out/roofline.json`, or `PARAMS.roofline`) and its Amdahl ceiling, with a note when the
ceiling is below the band; and, when the legs' serving logs are found (`server_log`, else
`server.log` beside the leg's receipt), the decode step time by batch size
([decode_step.py](decode_step.py), sglang "Decode batch" lines). `PARAMS.max_decode_step_regression_pct`
turns that table into a gate: a candidate whose median step at some batch size is slower than
stock by more than that percentage is judged `regression` even inside the throughput band.

The call writes `ab.json`, `REPORT.md` and `result.json`. Set `OUT`, `PARAMS`, `STAGE=evaluate`
and `STAGE_SKILL=native-e2e` for a direct call. The standard applyback finalizer first
checks kernel integration, then calls this same evaluator with that verified metadata;
it keeps its existing output envelope and does not calculate a second verdict.

`ok` means the comparison is admissible, not that the candidate won or met the task's
performance floor. The controller reads all required objectives and the uncertainty
policy before adoption. Preserve the reference when the candidate is worse or inconclusive.
