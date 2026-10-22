# Endpointing on real phone calls: silence timeout vs a small causal model

Local only. SpokenWOZ test (CC BY-NC 4.0). Not for publication. Reproduce: `run.py` (stage 1 cached in `cache/`), then `smartturn.py`.
Raw output: `out.txt`, `out_smartturn.txt`, `results.json`, `smartturn_results.json`. Plot: `curve.png`.

## Bottom line

**The model does not beat the silence timeout in any way that matters.** At the same cut-off rate as a 0.8 s timeout (30.8% of HOLD pauses), the best model answers at 0.78 s instead of 0.80 s (gboost, fallback 0.8 s; paired bootstrap 95% CI 0.01 to 0.02 s saving). That is noise-level for a voice agent. Where the cascade has a longer fallback (1.2 s) the 95% CI of the saving spans zero (-0.11 to +0.02 s). At the same latency, the cut-off rate does not drop either. A small gain shows up only in the 0.3 to 0.5 s regime (about 6 to 9 points fewer cut-offs at equal wait, point estimates, no CI computed for that table). The open smart-turn v3.2 model, run zero-shot, is about equal to the timeout above 0.5 s and slightly better below it, and worse at 0.8 s and above.

Why: with 8 kHz crowd-sourced calls, the prosody features carry little. Held-out AUC for END vs HOLD is 0.71 to 0.73 (gboost) and 0.65 to 0.71 (logreg). Most of that AUC comes from `d` (the pause length so far) and `mean_prev_pause` (a speaker-style feature). Pitch, energy slope and speech rate add little. Permutation importance (AUC drop): mean_prev_pause 0.065, d 0.063, f0_voiced 0.034, pause_e_rel 0.017, f0_rel 0.015, e_slope 0.012, others <= 0.009.

## Data and labels

- 1000 test dialogues available. Excluded all 69 dialogue ids in `data/real_tel/truth` and `data/real_tel_holdout/truth`. Split the rest by dialogue id with seed 11 into 200 train and 100 test (duration <= 400 s). 6 calls failed a channel sanity check (agent turns not louder on ch1) and were dropped, leaving 195 train and 99 test.
- Audio: ffmpeg decode (handles float wav), resample to 16 kHz, caller channel ch0. Silero VAD (`callcheck.vad.SileroVAD`, default settings) gives caller speech segments. A pause is a VAD gap of >= 0.2 s after a caller segment.
- Annotation is used only for the label and for the agent's own floor. Agent turn times are padded-corrected (+0.34 s start, -0.35 s end, as in `build_real_tel.py`). Agent turns of <= 2 words count as backchannel and are ignored.
- Label END: the next non-backchannel agent turn starts before the caller's next VAD segment (or the call ends). HOLD: the caller resumes first.
- Dropped pauses: the agent already holds the floor at pause start; the agent starts less than 0.2 s into the pause; the last pause with no agent turn after it.
- Result: 5008 train pauses (55.8% END), 2328 test pauses (61.6% END).

## Features (causal: audio up to t = pause start + d, plus the agent's own speech timing)

d, last segment duration, f0 slope over the last 300 ms of speech (simple FFT autocorrelation, 40 ms frames), f0 level vs segment median, voiced fraction, energy slope over last 300 ms (dB/s), energy of last 300 ms vs segment mean, speech-rate proxy (envelope peaks per second, last 1 s), turn length so far, number of pauses so far in the turn, mean previous pause length in the turn, mean pause energy so far vs speech. "Turn start" uses the agent's last non-backchannel turn end (the agent knows its own audio). One model across d in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5} (`d` is a feature). Models: StandardScaler + LogisticRegression; HistGradientBoosting (depth 3, 150 iters).

## Policies and metrics (test split, 99 calls, 2328 pauses)

- Silence timeout T (0.2 to 1.5 s): fire when the silence reaches T.
- Model: check at the decision delays (spec4 = {0.2, 0.3, 0.5, 0.8}; dense = all 10 delays). Fire at the first delay where P(END) >= threshold. If nothing fires, force a fire at fallback F (0.8, 1.2 or 1.5 s). The plot uses F = 1.2.
- Cut-off rate = % of all HOLD pauses where the agent fires before the caller resumes. Latency = added wait on END pauses (time from pause start to the fire; if nothing fires, F). Mean latency is the main axis because model latencies are discrete; median is in `results.json`.
- Bootstrap: resample dialogues (1000 for timeout, 300 for model curves; same draws across policies so comparisons are paired).

## Numbers

Silence timeout (cut-off % with 95% CI; mean wait = T):

| T (s) | cut-off % | 95% CI |
|---|---|---|
| 0.3 | 80.4 | 77.0 to 83.6 |
| 0.5 | 49.4 | 44.6 to 53.8 |
| 0.8 | 30.8 | 26.6 to 35.3 |
| 1.0 | 26.1 | 21.7 to 30.7 |
| 1.2 | 24.5 | 20.2 to 28.8 |
| 1.5 | 21.6 | 17.5 to 25.5 |

Minimum cut-off % reachable at mean added wait <= L (point estimates; model fallback F = 1.2):

| L (s) | timeout | logreg | gboost | gboost dense | smart-turn |
|---|---|---|---|---|---|
| 0.3 | 80.4 | 82.9 | 71.2 | 70.2 | 65.7 |
| 0.4 | 62.3 | 64.7 | 56.6 | 56.1 | 55.2 |
| 0.5 | 49.4 | 58.7 | 47.8 | 48.4 | 46.7 |
| 0.6 | 40.4 | 47.2 | 40.8 | 39.4 | 39.9 |
| 0.8 | 30.8 | 37.8 | 31.5 | 33.4 | 34.7 |
| 1.0 | 26.1 | 28.2 | 26.3 | 27.5 | 34.7 |

The 0.8 s headline (paired bootstrap, 95% CI of the saving; positive = model better):

| model | latency saving at equal cut-off (s) | cut-off saving at equal latency (points) |
|---|---|---|
| gboost spec4, F=0.8 | +0.018 (0.014 to 0.023) | 0.0 (0 to 0) |
| gboost spec4, F=1.2 | -0.078 (-0.108 to +0.019) | -1.3 (-4.4 to +1.2) |
| gboost dense, F=1.2 | -0.041 (-0.136 to +0.001) | -2.4 (-5.0 to +0.1) |
| logreg spec4, F=0.8 | +0.003 (0.000 to 0.010) | 0.0 (0.0 to 0.0) |
| logreg spec4, F=1.2 | -0.111 (-0.204 to -0.064) | -6.2 (-9.8 to -2.0) |

(Negative saving here comes from the longer fallback; the F=0.8 rows are the like-for-like comparison with a 0.8 s timeout. Full table in `results.json`, key `headline`.)

Reference: in these calls the human operator starts speaking a median 1.00 s (IQR 0.76 to 1.31 s) after the caller's VAD pause start on END pauses. A 0.8 s timeout is already faster than the human agents in the data.

Smart-turn v3.2 (pipecat-ai, int8 ONNX, Whisper-tiny encoder, last 8 s of the caller channel up to t, same decision points): AUC 0.73 at d=0.2, 0.72 at 0.3, 0.69 at 0.5, 0.63 at 0.8, 0.60 at 1.2. It falls with d because more of the 8 s window is silence. Its probabilities are nearly saturated: thresholds from 0.50 to 0.71 sweep the whole 100% to 44% cut-off range in a narrow step, so the curve is coarse.

## Limits (read before quoting any number)

- The callers and the "agents" are crowd-sourced human operators, not bots. Human operators leave long, slow gaps (median 1.0 s) and a quarter of HOLD pauses last over 1.2 s (callers read details, think, spell numbers). A bot agent faces a different pause mix, and this data says nothing about how callers behave when they know it is a bot.
- 8 kHz telephone audio upsampled to 16 kHz. f0 and spectral cues are weaker than in wideband audio. Smart-turn was trained on wideband speech, so its numbers are a domain-mismatch baseline, not its best case.
- Annotation pad: the annotated boundaries sit about 0.34 s before voice onset and 0.35 s after voice offset. We subtracted a constant pad. The residual error (IQR about 0.1 s) changes labels for END/HOLD at the 0.2 s margin and blurs the human-gap reference.
- Labels are noisy: a HOLD may be a pause where the human agent also would have waited, an END may be one where the operator was slow. Labels reflect what happened, not what was best.
- 99 test calls, 2328 pauses. Bootstrap CIs are by dialogue but the model-vs-timeout curve comparison at 0.3 to 0.6 s has no CI here. The `mean_prev_pause` feature is a per-speaker style feature and may inflate AUC; no speaker-disjoint check was done beyond dialogue split.
- Models trained on 195 calls only; a bigger corpus or a learned audio encoder may do better. No tuning beyond one default setting each. Threshold sweeps are shown on the test set as a curve, not chosen on it.
- Speech rate uses an envelope proxy, no transcript.
- `cache/` (1.5 GB, includes decoded audio) and `models/` stay local; they derive from CC BY-NC data.
