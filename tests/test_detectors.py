"""Event rules on hand-built segment lists, so each rule is checked without audio or a VAD."""
from callcheck.analyze import find_cutoffs, find_false_barge_ins


def test_talk_over():
    # Caller talks 10-14 s without pausing; the agent starts at 12 s.
    ev = find_cutoffs([(10.0, 14.0)], [(12.0, 15.0)])
    assert [e["kind"] for e in ev] == ["talk_over"]


def test_pause_jump():
    # Caller pauses 0.5 s, agent jumps in, caller resumes 0.6 s later with a real phrase.
    ev = find_cutoffs([(10.0, 12.0), (13.1, 15.0)], [(12.5, 15.5)])
    assert [e["kind"] for e in ev] == ["pause_jump"]


def test_normal_reply_is_not_a_cutoff():
    # Caller finishes, agent answers 0.6 s later and the caller stays quiet.
    assert find_cutoffs([(10.0, 12.0)], [(12.6, 16.0)]) == []


def test_early_answer_is_not_a_cutoff():
    # Caller replies 3 s into a long agent turn: answering early, not being cut off.
    assert find_cutoffs([(5.0, 8.0), (13.0, 14.0)], [(8.6, 15.0)]) == []


def test_backchannel_does_not_count_as_resuming():
    # A 0.3 s "mm-hm" after the agent starts is too short to be the caller resuming.
    assert find_cutoffs([(10.0, 12.0), (12.9, 13.2)], [(12.5, 16.0)]) == []


def test_cough_stops_agent():
    agent = [(20.0, 24.3), (27.0, 28.0)]
    ev = find_false_barge_ins([(24.0, 24.25)], [], agent)
    assert len(ev) == 1 and ev[0]["agent_stopped_at"] == 24.3


def test_cough_agent_keeps_talking():
    # The agent talks straight through the sound.
    assert find_false_barge_ins([(22.0, 22.25)], [], [(20.0, 26.0)]) == []


def test_cough_in_breath_and_agent_never_resumes():
    ev = find_false_barge_ins([(24.1, 24.35)], [], [(20.0, 24.0)])
    assert len(ev) == 1


def test_cough_in_breath_agent_resumes():
    # Same sound in a breath, but the agent carries on for 3 s: not a barge-in.
    assert find_false_barge_ins([(24.1, 24.35)], [], [(20.0, 24.0), (24.2, 27.5)]) == []


def test_word_is_not_a_cough():
    assert find_false_barge_ins([(24.0, 24.25)], [(23.9, 24.3)], [(20.0, 24.3)]) == []


def test_fragments_reported_once():
    ev = find_false_barge_ins([(24.0, 24.1), (24.2, 24.3)], [], [(20.0, 24.4), (27.0, 28.0)])
    assert len(ev) == 1


def test_sound_segments_ignore_steady_noise():
    """Steady background noise above the fixed threshold must not read as one long sound."""
    import numpy as np
    from callcheck.vad import SR, sound_segments
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * 4) * 0.03).astype(np.float32)  # noise louder than rms_on=0.02
    audio[SR : SR + SR // 2] += np.sin(np.arange(SR // 2) * 2 * np.pi * 300 / SR).astype(np.float32) * 0.5
    segs = sound_segments(audio)
    assert len(segs) == 1 and abs(segs[0][0] - 1.0) < 0.05 and abs(segs[0][1] - 1.5) < 0.05
