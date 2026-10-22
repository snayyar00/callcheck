# callcheck

Grade voice-agent call recordings for the turn-taking failures callers notice:

- **slow_response**: the agent took more than 1.5 s to answer
- **dead_air**: more than 3 s
- **cutoff**: the agent started talking while the caller was still going
- **false_barge_in**: a cough or bump on the line made the agent stop mid-sentence
- **wrong_readback**: the agent read back a different phone number than the caller said (optional, needs speech-to-text)

It works on a two-channel recording (caller on one channel, agent on the other) from any platform. No vendor API, no call IDs. Two halves of onboarding a customer onto a voice agent: before, read their call transcripts and work out what can be automated (`audit/`); after, grade the agent's real calls for the failures callers notice (`callcheck`).

```
pip install -e . && callcheck call.wav
```

The interesting part is not the tool. It is how far you can trust it, which is what most of this page is about.

## How it works

Silero VAD (ONNX) on each channel, with an energy detector as a fallback for voices Silero under-detects. Caller pieces are joined into turns. Latency is measured from the end of the caller's turn to the start of the agent's. A cut-off is either the agent starting while caller speech is still running, or the agent jumping into a short caller pause (under 1 s) after which the caller resumes within 1.5 s. A false barge-in is a short non-speech sound on the caller line followed by the agent going quiet within 1 s. The readback check runs faster-whisper on both channels and compares digit strings.

All thresholds are in `callcheck/analyze.py:Config`.

## How well it works

Every number below comes from a script in this repo. Sets marked *untouched* were not looked at while tuning.

### Synthetic calls with planted faults (exact ground truth)

60 calls, seed 2026, *untouched*. Phone copy is 8 kHz mu-law.

| Fault | Planted | Found (phone / wideband) | False alarms |
|---|---|---|---|
| slow_response | 21 | 21 / 21 | 0 / 0 |
| dead_air | 16 | 16 / 16 | 0 / 0 |
| cutoff | 17 | 17 / 17 | 0 / 0 |
| false_barge_in | 22 | 22 / 22 | 0 / 0 |
| wrong_readback | 15 | 15 / 13 | 0 / 1 |

Latency error per turn: median 51 ms, max 226 ms (337 turns). The same set with real human callers (LibriSpeech voices) finds everything too, but latency error reaches 551 ms because real speech trails off.

This is an upper bound. Synthetic calls are clean and the faults are planted the way I expected them.

### Real phone calls, human-annotated timing

SpokenWOZ test set: real two-party customer-service phone calls, 8 kHz stereo, one speaker per channel, utterance times marked by annotators. The annotation pads every utterance by about 0.35 s on each side (measured against channel energy over 1090 turns), so that constant is removed before scoring.

Tuned on 32 calls, scored on 40 *untouched* calls (131 min):

| Check | Result |
|---|---|
| Response latency, 731 turns | median error -0.08 s, 90% within 0.21 s |
| Gaps over 1.5 s | precision 85%, recall 78% |
| Gaps over 3 s | precision 64%, recall 82% |
| Cut-off (agent talks over caller) | precision 57%, recall 39% |

The cut-off detector in the first version scored precision 13%, recall 14% on the first 37 of these calls. It counted line clicks as the caller resuming, and it missed every case where the caller never paused.

### Real meetings, word-level timing (harder than phone)

AMI meeting corpus, 29 four-minute windows of two people talking, individual headset mics. Many short "yeah" / "mm-hm" acknowledgements, laughter, and voice bleed between mics.

| Check | All 29 clips | 19 clips with little bleed |
|---|---|---|
| Latency within 0.5 s | 78% | |
| Cut-off | precision 43%, recall 27% | precision 51%, recall 33% |

## Before the migration: transcript audit

`audit/run.py` is the step before any agent exists. Give it a customer's call transcripts and it reports what people call about, which call types a voice agent could handle end to end, which back-office systems each one needs, and a savings estimate with every assumption printed. One LLM pass per conversation (Claude Sonnet via Vercel AI Gateway), then one pass to group the reasons into call types.

Tested on ABCD (ASAPP, MIT licence): human customer-service conversations labelled with what the customer wanted and every system action the human agent took. The model sees only the spoken lines, as in a real transcript, and is scored against the hidden labels. Prompts were adjusted on 40 dev conversations; the numbers below are from one run on 300 *untouched* test conversations (about $1 of API calls). Full output: `audit/SAMPLE_REPORT.md`.

| Check | Result |
|---|---|
| Systems an agent would need (account lookup, identity check, orders, refunds, ...) | precision 71%, recall 89% |
| Call types found vs the 10 true top-level reasons | 12 types, purity 70%, NMI 0.66 |
| Calls the human agent escalated: audit said "partial" | 12 of 22 (vs 5% of other calls) |

It leans toward listing a system when unsure, which is the safer mistake when scoping an integration. It is too optimistic about automation: 10 of the 22 calls a human escalated were still marked fully automatable. Treat its automation split as a first pass for a person to review, not a number to quote.

## Side study: can audio alone tell when the caller is done?

The same real calls give a clean test of endpointing, the decision every voice agent makes at each caller pause. On 99 held-out SpokenWOZ calls (2328 caller pauses), I compared a plain silence timeout with a gradient-boosted model on prosody features and with the open smart-turn v3.2 model run zero-shot.

| Wait before answering | Silence timeout cuts in on | Best model |
|---|---|---|
| 0.3 s | 80% of mid-turn pauses | 66% (smart-turn) |
| 0.5 s | 49% | 47% |
| 0.8 s | 31% | 31% (no gain) |

From audio alone, no model beat the timeout at the waits a production agent would use. It only helps when you answer very fast. Most of what the model learned was pause length and how long this caller usually pauses. Pitch and energy added little. Human operators in these calls answer a median 1.0 s after the caller stops. Details, confidence intervals and limits are in `research/endpointing/RESULTS.md`. The likely next step is text: what the caller said, not how it sounded.

## Where it breaks

- **Echo / bleed.** If the agent's voice leaks into the caller channel (or the reverse), callcheck sees phantom turns. On clips where the other channel is near the same level, almost every caller sentence is flagged as a cut-off.
- **Backchannels.** A caller "mm-hm" is not a turn, but callcheck has no class for it yet, so it can reset the latency timer.
- **Talk-over recall.** It still misses most interruptions where the caller keeps talking and never pauses.
- **Laughs and coughs on real lines.** It finds planted coughs, but on AMI it flagged 0 of 25 real laughs or coughs that stopped the other speaker.
- **Mono recordings** are refused. Splitting a mono mix by speaker loses the overlaps that matter most.

## Reproduce

```
uv run --python 3.12 --with numpy python synth/generate.py --calls 60 --seed 2026 --out data/final
uv run --python 3.12 --with numpy python synth/verify.py data/final
uv run --python 3.12 --with numpy --with onnxruntime python score.py data/final phone
uv run --python 3.12 --with numpy --with onnxruntime --with faster-whisper python score_entities.py data/final phone
mkdir -p data/.corpora/spokenwoz_text data/.corpora/spokenwoz_test
curl -L -o data/.corpora/spokenwoz_text/data.json https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed/resolve/main/data.json
curl -L -o data/.corpora/spokenwoz_test/audio_5700_test.zip https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Audio-Fixed/resolve/main/audio_5700_test.zip
uv run --python 3.12 --with numpy python realdata/telephone/build_real_tel.py           # 32 tuning calls
uv run --python 3.12 --with numpy python realdata/telephone/build_real_tel.py holdout   # 40 held-out calls
uv run --python 3.12 --with numpy --with onnxruntime python realdata/telephone/score_real.py holdout
sh realdata/ami/fetch.sh                                   # ~2.2 GB of AMI audio
uv run --python 3.12 --with numpy python realdata/ami/build_pairs.py
uv run --python 3.12 --with numpy --with onnxruntime python realdata/ami/score_ami.py
```

`synth/verify.py` checks every synthetic label against the audio with a plain energy detector, independent of callcheck. Breaking a label on purpose makes it fail.

## Data and licences

Audio is not in the repo. The scripts download or generate it.

- Synthetic calls: macOS `say` voices, generated by `synth/generate.py`.
- Human caller voices: LibriSpeech (CC BY 4.0).
- SpokenWOZ (CC BY-NC 4.0): used for evaluation only; audio and derived labels stay local.
- AMI Meeting Corpus (CC BY 4.0), University of Edinburgh.
- ABCD (MIT), ASAPP Research; download `abcd_v1.1.json.gz` from github.com/asappresearch/abcd into `data/.corpora/abcd/`.
- Silero VAD (MIT), downloaded on first run.

Code: MIT.
