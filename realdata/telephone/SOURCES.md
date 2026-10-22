# Real two-party telephone corpora: survey (2026-10-22)

| Corpus | Real phone? | Per-speaker channels | Timing annotation | Free, no login | Licence | Verdict |
|---|---|---|---|---|---|---|
| **SpokenWOZ** (test set) https://spokenwoz.github.io/SpokenWOZ-github.io/ ; audio https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Audio-Fixed (2.18 GB zip, 1000 calls) ; text https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Text-Fixed (52 MB `data.json`) | Human-human customer/agent calls over real voice calls (crowd workers, Cambridge-style task domains). 8 kHz | YES. Stereo WAV, ch0 = customer ("user"), ch1 = service ("system"). Verified by own-channel RMS: other channel is 100x quieter (RMS ~5-85 vs 1000-6000). Almost no cross-talk. | Per-utterance `BeginTime`/`EndTime` (ms) per word, but word times are evenly interpolated inside the utterance (e.g. "a" gets as long as "staff"). Only utterance start/end carry information, and they are padded (see README below). `ChannelId` field is unreliable (nearly all 0); use `tag` instead. | YES (HF, not gated; no account) | CC BY-NC 4.0 (stated on project page). Non-commercial: keep derived WAV/truth local, do not commit or ship. | **USED** |
| CABank CallHome English (TalkBank) https://talkbank.org/ca/access/CallHome/eng.html | Yes, 30 min US phone calls | Transcripts with bullet timestamps per turn; media is 2-channel | Turn-level | Transcripts yes; **media returns a TalkBank login modal** (`media.talkbank.org/ca/CallHome/eng/*.mp3` serves an auth script, not audio) | Page names no licence, citation required; underlying data is LDC97S42 (paid) | Skipped: media needs login |
| Santa Barbara Corpus (SBCSAE) https://openslr.org/155/ ; also https://ca.talkbank.org/access/SBCSAE.html | Mostly face-to-face, a few phone calls; 22.05 kHz | 60 stereo files, but channels are mic mixes, not one speaker per channel | Intonation-unit time marks | Yes (6.1 GB) | CC BY-ND 3.0 US | Skipped: not telephone, no per-speaker separation, ND |
| Fisher English / Switchboard / CallFriend / CallHome audio (LDC) https://catalog.ldc.upenn.edu | Yes | Yes | Yes | **No, paid/LDC membership** | LDC | Skipped (paid) |
| AMI (already in data/.corpora) | Meetings, not telephone | Headset channels | Yes | Yes | CC BY 4.0 | Not telephone; not used here |

No openslr.org telephone-dialogue set with per-speaker channels was found in the search (searched by name; not an exhaustive index crawl).

## What the data actually is
- Calls: 87.7 min, 32 calls, ids in `data/real_tel/manifest.json`. 8 kHz stereo upsampled to 16 kHz by ffmpeg. Upsampling adds no bandwidth, so this is real narrowband telephone audio at the 16 kHz container.
- Speaker mapping: customer = `caller` (ch0), service operator = `agent` (ch1).
- Caveat: operators are human crowd workers, not bots, so "response latency" is human latency; the point is timing realism, not agent behaviour.

## How truth is derived (`realdata/telephone/build_real_tel.py`)
1. One turn per annotated dialogue-log entry (consecutive entries by the same speaker are merged). Start = first word `BeginTime`, end = last word `EndTime`. These are the only human-controlled boundaries; interior word times are interpolated and not used.
2. **Annotation bias found and measured.** Against each speaker's own (cross-talk-free) channel energy over 1090 turns: annotated start is a median 0.34 s BEFORE voice onset (IQR 0.29-0.35); annotated end is a median 0.35 s AFTER voice offset (IQR 0.32-0.39). Raw latencies are therefore ~0.7 s too small if taken literally. Truth fields come in two variants:
   - primary (`turns`, `latencies`, `overlaps`, `silences_gt3s`): constant pad removed (start +0.34, end -0.35). Still human-annotation-derived (no VAD); the two constants are one global calibration from channel energy, not per-turn snapping. After correction the median onset error is 0.01-0.02 s, p90 0.3-0.4 s.
   - `*_raw`: annotation as published.
3. Response latency: agent turn whose previous turn in the log is the caller and caller_end <= agent_start; latency = agent_start - caller_end.
4. Overlap (cut-off candidate): agent_start lies inside a caller turn with >= 0.3 s before and >= 0.3 s after. Flagged `backchannel` when the agent turn has <= 2 words.
5. Silence > 3 s: gap with neither speaker annotated. Gap thresholds 1.5 s and 3 s for slow/dead air are applied to response latencies.
6. Precision limits: utterance-level only, ~0.3 s p90 residual boundary error; annotators often truncate or omit short or trailing caller speech (see report), so truth is itself noisy at the ~5% level for latency and noisier for overlaps. Only 17 overlap events exist in 32 calls; treat cut-off precision/recall as anecdotal.
7. Sample selection: valid dialogues 60-300 s with >= 12 turns; 10 chosen as "hardest" by count of overlaps + >3 s silences + >1.5 s gaps, the rest random (seed 7). Calls failing a channel-sanity check (caller louder on ch0, agent on ch1) are rejected (3 rejected).
