# Human caller speech sources

| Corpus | URL | Licence | Used for |
|---|---|---|---|
| LibriSpeech dev-clean (40 speakers, 20 F / 20 M, 337 MB) | https://www.openslr.org/resources/12/dev-clean.tar.gz (page: https://www.openslr.org/12) | CC BY 4.0 (Panayotov, Chen, Povey, Khudanpur, ICASSP 2015) | Caller turns in `data/human` (synth/generate_human.py) |

No account is needed. Extract to `data/.corpora/ls/` (so the files are at
`data/.corpora/ls/LibriSpeech/dev-clean/<spk>/<chap>/*.flac`).

Limits: read audiobook speech, not conversation. It is clean, close-mic, 16 kHz, with no
overlap, no backchannels and no line noise. Utterance text is unrelated to the agent's
questions, so no entity ground truth is written (`entities` is `{}`).

Not used: Mozilla Common Voice (needs an account). AMI (CC BY 4.0) was not used because it
needs per-meeting downloads and a VAD pass to cut turns.
