# callcheck

Grade voice-agent call recordings for the failures callers notice: slow responses, dead air, cutting the caller off, stopping for a cough, and reading back the wrong phone number. Works on recordings from any platform.

Status: synthetic test set done. Analyzer next.

## Synthetic test set

24 two-channel calls (left = caller, right = agent) with faults planted at known times, so the analyzer can be scored against ground truth.

```
uv run --python 3.12 --with numpy python synth/generate.py --calls 24 --seed 7 --out data/synthetic
uv run --python 3.12 --with numpy python synth/verify.py data/synthetic
```

| Fault | Calls | How it is planted |
|---|---|---|
| cutoff | 5 | agent starts talking 0.35-0.6 s into a mid-sentence caller pause |
| slow_response | 9 | agent answers after 1.7-2.6 s (threshold 1.5 s) |
| dead_air | 7 | agent answers after 3.2-5.0 s (threshold 3.0 s) |
| false_barge_in | 11 | a 250 ms cough on the caller line stops the agent 300 ms later |
| wrong_readback | 12 | one digit of the phone number changed in the agent's readback |
| clean | 4 | no faults, to measure false alarms |

Each call has a wideband WAV (16 kHz), a phone-line copy (300-3400 Hz, 8 kHz mu-law) and a ground-truth JSON with every turn, every planted fault and the phone number said vs read back.

Two checks keep the labels honest:
- `generate.py` refuses to write a call if any response latency disagrees with its label.
- `verify.py` reads the audio back with a plain energy detector, independent of the analyzer, and checks every turn boundary, that the only overlaps are planted ones, and the phone-copy format. Breaking a label on purpose makes it fail.

Limits: voices are macOS `say`, which is cleaner and more regular than real callers. Planted cut-offs keep the caller talking through the interruption. Real calls will be noisier, so results on this set are an upper bound.
