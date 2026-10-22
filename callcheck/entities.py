"""Pull the caller's phone number and the agent's read-back out of two transcripts.

Emits a `wrong_readback` event when the agent's first read-back of the caller's
number differs from what the caller said, and `time_to_info_s`: seconds from call
start until the agent read the number back correctly and the caller's next turn
(the confirmation) ended.
"""

import re

DIGITS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
# Sound-alikes that STT writes for digit words. Only used inside a run of digits.
NEAR = {"to": "2", "too": "2", "for": "4", "fore": "4", "won": "1", "ate": "8", "age": "8", "sex": "6", "tree": "3"}
PHONE_LEN = 10
AGENT_TURN_GAP_S = 1.0
NUMBER_GAP_S = 4.0   # digit runs further apart than this are not one phone number
CALLER_TURN_GAP_S = 1.0   # pauses shorter than this stay in one turn


def _token_digits(word, strict):
    """Digits for one transcript word, or None. strict=False also accepts sound-alikes."""
    w = re.sub(r"[^\w-]", "", word.lower())
    if not w:
        return ""   # pure punctuation: transparent
    if w in DIGITS:
        return DIGITS[w]
    if re.fullmatch(r"[\d-]+", w):
        return w.replace("-", "")
    if re.fullmatch(r"a\d+", w):   # "eight" heard as "A": "A20" is 820
        return "8" + w[1:]
    if not strict and w in NEAR:
        return NEAR[w]
    return None


def _flat_words(segments):
    words = []
    for si, seg in enumerate(segments):
        for w in seg["words"]:
            words.append({**w, "seg": si})
    return words


def find_numbers(segments):
    """Runs of spoken digits: [{"digits", "start", "end", "seg"}], in time order.

    A run may cross segment boundaries. "oh" and sound-alikes such as "to" only
    count when they sit next to a real digit word.
    """
    words = _flat_words(segments)
    kinds = []   # per word: digits string or None; 'weak' flag for ambiguous words
    for w in words:
        strict = _token_digits(w["word"], True)
        loose = _token_digits(w["word"], False)
        weak = strict is None and loose is not None or re.sub(r"[^\w]", "", w["word"].lower()) in ("o", "oh")
        kinds.append((strict if strict is not None else loose, weak))
    runs, cur = [], []
    for w, (d, weak) in zip(words, kinds):
        if d is None:
            if cur:
                runs.append(cur)
            cur = []
        elif d != "":
            cur.append((w, d, weak))
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        # Weak words ("oh", "to", "for") at the edges are noise unless the run is a digit short of a phone number.
        lo, hi = 0, len(run)
        while lo < hi and run[lo][2]:
            lo += 1
        while hi > lo and run[hi - 1][2]:
            hi -= 1
        if lo == hi:
            continue
        count = sum(len(d) for _, d, _ in run[lo:hi])
        while count < PHONE_LEN and lo > 0:
            lo -= 1
            count += len(run[lo][1])
        while count < PHONE_LEN and hi < len(run):
            count += len(run[hi][1])
            hi += 1
        run = run[lo:hi]
        digits = "".join(d for _, d, _ in run)
        out.append({"digits": digits, "start": run[0][0]["start"], "end": run[-1][0]["end"],
                    "seg": run[0][0]["seg"]})
    return out


def _phone_runs(segments, said=None):
    """Runs of at least 10 digits, trimmed to the first 10 (a stray digit after the number
    is usually the next word misheard). A longer run that holds `said` as a subsequence
    is that number with a digit or two of STT noise, so it reads as `said`."""
    out = []
    for r in find_numbers(segments):
        d = r["digits"]
        if len(d) < PHONE_LEN:
            continue
        if said and len(d) > PHONE_LEN and _is_subsequence(said, d):
            out.append({**r, "digits": said})
        else:
            out.append({**r, "digits": d[:PHONE_LEN]})
    return out


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _is_subsequence(short, long):
    it = iter(long)
    return all(ch in it for ch in short)


def _restore_repeat(run, said):
    """STT often drops one of two equal digits ("two two" heard as "two"), leaving 9 digits.
    Put back the repeat that brings the run closest to `said`; give up if it is more than one digit off."""
    best = None
    for i in range(len(run)):
        cand = run[:i + 1] + run[i] + run[i + 1:]
        h = _hamming(cand, said)
        if best is None or h < best[0]:
            best = (h, cand)
    return best[1] if best[0] <= 1 else None


def _nine_digit_runs(segments, said):
    """Nine-digit runs that look like `said` with a swallowed repeated digit, as 10-digit readings."""
    found = []
    for r in find_numbers(segments):
        if len(r["digits"]) == PHONE_LEN - 1:
            read = _restore_repeat(r["digits"], said)
            if read:
                found.append({**r, "digits": read})
    return found


def _merge(spans, gap_s):
    """Join (start, end) spans closer than gap_s."""
    turns = []
    for s, e in sorted(spans):
        if turns and s - turns[-1][1] < gap_s:
            turns[-1][1] = max(turns[-1][1], e)
        else:
            turns.append([s, e])
    return [tuple(t) for t in turns]


def _turn_with(turns, start, end):
    """The turn that overlaps the span [start, end] most, or None."""
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(t[1], end) - max(t[0], start)
        if ov > best_ov:
            best, best_ov = t, ov
    return best


def _caller_number(runs, read_start, read_digits):
    """The number the caller said, from the digit runs that end before the agent's read-back.

    Caller STT is not exact either. A group heard twice or lost leaves 9 or 11+ digits.
    When the agent's read-back is a subsequence of the heard digits, or the other way round,
    the caller's number is taken to be the read-back (no fault is claimed on a doubt).
    Otherwise a later 10-digit repeat of the number ("No, it's ...") within 2 edits settles it.
    Returns (digits, start) or (None, None).
    """
    before = [r for r in runs if r["end"] <= read_start]
    # Keep the last cluster of runs. A digit heard in the caller's name sits seconds earlier.
    while len(before) > 1 and before[1]["start"] - before[0]["end"] > NUMBER_GAP_S:
        before = before[1:]
    if not before:
        return None, None
    heard = "".join(r["digits"] for r in before)
    start = before[0]["start"]
    if len(heard) == PHONE_LEN:
        return heard, start
    if len(heard) > PHONE_LEN:
        if read_digits and _is_subsequence(read_digits, heard):
            return read_digits, start
        return heard[:PHONE_LEN], start
    if len(heard) >= 6 and read_digits and _is_subsequence(heard, read_digits):
        return read_digits, start
    for r in runs:
        if r["start"] >= read_start and len(r["digits"]) >= PHONE_LEN and \
                _edit_distance(heard, r["digits"][:PHONE_LEN]) <= 2:
            return r["digits"][:PHONE_LEN], start
    return None, None


def extract(caller_segs, agent_segs, caller_speech=None, agent_speech=None):
    """Returns {"said", "read_back", "events", "time_to_info_s"}.

    caller_speech and agent_speech are optional VAD speech segments [(start, end)].
    Whisper segment times run about half a second early, so turn edges come from
    the VAD when it is given.
    """
    c_turns = _merge(caller_speech or [(s["start"], s["end"]) for s in caller_segs], CALLER_TURN_GAP_S)
    a_turns = _merge(agent_speech or [(s["start"], s["end"]) for s in agent_segs], AGENT_TURN_GAP_S)
    result = {"said": None, "read_back": None, "events": [], "time_to_info_s": None}
    c_runs = find_numbers(caller_segs)
    a_runs = [r for r in find_numbers(agent_segs) if len(r["digits"]) >= PHONE_LEN - 1]
    if not c_runs or not a_runs:
        return result
    # The agent's first read-back is its first long digit run after the caller's first digits.
    a_runs = [r for r in a_runs if r["end"] > c_runs[0]["start"]]
    if not a_runs:
        return result
    said, said_start = _caller_number(c_runs, a_runs[0]["start"], a_runs[0]["digits"][:PHONE_LEN])
    if said is None:
        return result
    result["said"] = said
    reads = _phone_runs(agent_segs, said) + _nine_digit_runs(agent_segs, said)
    reads = sorted((r for r in reads if r["start"] >= said_start), key=lambda r: r["start"])
    if not reads:
        return result
    result["read_back"] = reads[0]["digits"]
    if reads[0]["digits"] != said:
        turn = _turn_with(a_turns, reads[0]["start"], reads[0]["end"])
        result["events"].append({
            "type": "wrong_readback",
            "time": round(turn[0] if turn else reads[0]["start"], 3),
            "said": said, "read_back": reads[0]["digits"],
        })
    right = next((r for r in reads if r["digits"] == said), None)
    if right:
        turn = _turn_with(a_turns, right["start"], right["end"])
        after = turn[1] if turn else right["end"]
        confirm = next((t for t in c_turns if t[0] >= after - 0.1), None)
        if confirm:
            result["time_to_info_s"] = round(confirm[1], 3)
    return result
