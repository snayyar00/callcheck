# More results

Detail behind the README. Paths are relative to the repo root.

## Real meetings

The AMI meeting corpus (CC BY 4.0) gives 29 four-minute windows of two people on individual headset mics. Meetings are harder than phone calls: many short "yeah" and "mm-hm" replies, laughter, and voice bleed between mics. Latency is within 0.5 s on 78% of matched turns. Cut-off precision is 43% and recall 27%, up from 29% and 28% for the first version (`results/ami.txt`, `results/ami_first_version.txt`). Most of that comes from talk-overs. The pause-jump rule alone does badly on meetings: precision 5%, recall 12%.

## Synthetic calls with planted faults

60 synthetic calls with exact ground truth (seed 2026, generated once and not used for tuning). Every fault type is found with no false alarms on both the wideband and the 8 kHz phone copy: slow 21, dead air 16, cut-off 17, false barge-in 22. Median latency error is 52 ms (`results/synthetic_final_*.txt`). Readback finds 15 of 15 wrong numbers on phone audio, and 13 of 15 with 1 false alarm on wideband (`results/synthetic_final_readback.txt`). The same check with real human caller voices from LibriSpeech also finds every fault (`results/synthetic_human_callers_*.txt`). Synthetic calls are clean, so treat this as an upper bound.

## The same calls with noise, echo and codecs

`synth/harden.py` adds background noise (traffic, TV, fan, car at 5 to 20 dB SNR), echo of the agent into the caller line, quiet callers, clipping, Opus at 8 to 12 kbps and packet loss to the 60 synthetic calls. Timing does not change, so the same ground truth applies.

| 60 calls | Wideband recall / precision | Phone recall / precision |
|---|---|---|
| slow_response | 95% / 100% (was 62% / 93%) | 90% / 100% (was 76% / 100%) |
| dead_air | 88% / 100% (was 62% / 100%) | 94% / 100% (was 94% / 100%) |
| cutoff | 76% / 20% (was 76% / 8%) | 76% / 22% (was 76% / 14%) |
| false_barge_in | 50% / 92% (was 32% / 100%) | 68% / 94% (was 59% / 93%) |

Sources: `results/hard_wideband.txt` and `results/hard_phone.txt`; the "was" numbers are in the `*_first_version.txt` files next to them. The per-condition split is in `results/hard_by_condition_wideband.txt`.

The first version's energy fallback used a fixed loudness threshold, so steady noise read as one long caller turn. It now uses the larger of that threshold and 4 times the channel's noise floor (the 10th percentile of frame loudness). The factor 4 was picked on noisy copies of the dev calls (`data/synthetic` and `data/heldout`, hardened with a different seed), not on these 60. Clean and real-call scores did not change.

What still fails: TV noise (speech-like, so the speech detector hears it too: cut-off precision 7%) and echo (the agent's voice lands on the caller channel: cut-off recall 20%, false barge-in recall 0%). Each call has several conditions at once, so these rows overlap.

## can audio alone tell when the caller is done?

The same real calls give a clean test of endpointing, the decision a voice agent makes at every caller pause. On 99 held-out SpokenWOZ calls (2328 caller pauses), I compared a plain silence timeout with a gradient-boosted model on prosody features and with the open smart-turn v3.2 model run zero-shot.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="endpointing-dark.png">
  <img alt="Share of mid-turn pauses where the agent would cut in, by wait time. At 0.3 s: timeout 80%, prosody model 71%, smart-turn 66%. At 0.8 s: timeout 31%, prosody model 32%, smart-turn 35%." src="endpointing-light.png" width="720">
</picture>

The models only help when the agent answers very fast. At the waits a production agent would use, the plain timeout does as well. Most of what the model learned was pause length and how long this caller usually pauses. Pitch and energy added little. Human operators in these calls answer a median 1.0 s after the caller stops. Details, confidence intervals and limits are in `research/endpointing/RESULTS.md`. The next thing to try is adding what the caller said, from the transcript, to the audio features.
