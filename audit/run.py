"""Transcript audit: what are customers calling about, and what could a voice agent handle?

Reads customer-service transcripts, asks an LLM per conversation for the reason, whether an
agent could resolve it end to end, and which back-office systems it would need. A second pass
groups the free-text reasons into a short list of call types. Then it writes a report with
volumes, automation potential and a savings estimate whose assumptions are all printed.

Evaluated on ABCD (ASAPP, MIT): ~10k human customer-service conversations labelled with the
customer's intent and every system action the human agent took. The LLM sees only the spoken
lines (action lines removed), the same as a real call transcript, and is scored against those
hidden labels.

    export AI_GATEWAY_API_KEY=...        # Vercel AI Gateway
    python audit/run.py --split dev  --n 40     # prompt iteration
    python audit/run.py --split test --n 300    # the scored run
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABCD = ROOT / "data/.corpora/abcd/abcd_v1.1.json.gz"
OUT = ROOT / "audit/out"
MODEL = os.environ.get("AI_GATEWAY_MODEL", "deepseek/deepseek-v4.1-flash")
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"

# Back-office systems an agent may need. ABCD's own action names map onto them, which is the truth.
SYSTEMS = {
    "account_lookup": "find the customer's account (name, email, phone, account id)",
    "identity_verification": "confirm identity with extra details beyond the name (e.g. account id + order id, or email/phone/zip) before acting on an order or account",
    "order_system": "look up, validate, change or place an order; shipping status",
    "refunds": "issue a refund or credit",
    "account_changes": "change stored account data: address, name, password, 2FA, or send a reset link",
    "membership_billing": "membership level, subscriptions, billing, promo codes",
    "knowledge_base": "answer a policy / FAQ / product question, or walk through troubleshooting steps, from documentation",
    "human_handoff": "escalate to a person or another team",
}
ACTION_TO_SYSTEM = {
    "pull-up-account": "account_lookup",
    "verify-identity": "identity_verification",
    "validate-purchase": "order_system", "update-order": "order_system", "shipping-status": "order_system",
    "make-purchase": "order_system",
    "offer-refund": "refunds",
    "update-account": "account_changes", "make-password": "account_changes", "send-link": "account_changes",
    "membership": "membership_billing", "subscription-status": "membership_billing", "promo-code": "membership_billing",
    "notify-team": "human_handoff",
}

EXTRACT_PROMPT = """You are auditing a company's customer-service conversations to plan a voice AI agent.
Read the conversation and answer in JSON only, no prose:
{
 "reason": "why the customer reached out, 3-8 words, generic (no names, no order numbers)",
 "automatable": "full" | "partial" | "no",
 "why": "one sentence: what makes it easy or hard for an AI agent",
 "systems": [subset of %s],
 "info_needed": ["details the agent had to collect, e.g. order id, email"]
}
Judge from what actually happened in this conversation, not from the topic in general:
"full" = an AI agent with the listed systems could have done everything the human agent did here.
"partial" = most of it, but one step needed a person: an exception to policy, an approval, a report
  to another team, a promise of follow-up, or calming an upset customer before anything could happen.
"no" = the core of the request needed a person: a complaint or dispute about a decision, a problem
  nobody on the call could fix, a negotiation, or the human agent had to escalate or transfer it.
If the human agent escalated, transferred, or said another team must handle it, it is not "full".
List a system only if resolving this specific request needs it.
System descriptions: %s

Conversation:
%s"""

TAXONOMY_PROMPT = """Here are call reasons from a company's customer-service audit, one per line with an id.
Group them into at most %d call types a business owner would recognise (e.g. "Order status",
"Password or login problem"). Each type gets a short name. Answer in JSON only:
{"types": [{"name": "...", "ids": [ids]}]}
Every id must appear exactly once.

%s"""


def llm(prompt, max_tokens=4000, timeout=120):  # reasoning models spend tokens thinking before the JSON
    key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
    if not key:
        sys.exit("set AI_GATEWAY_API_KEY (Vercel AI Gateway)")
    body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            text = r["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0)), r.get("usage", {})
        except Exception as e:  # rate limits, provider blips, a malformed reply
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def transcript(convo):
    """Spoken lines only. Action lines are the hidden labels, so the model must not see them."""
    return "\n".join(f"{who}: {text}" for who, text in convo["original"] if who in ("agent", "customer"))


def truth(convo):
    acts = {t["targets"][2] for t in convo["delexed"] if t["speaker"] == "action" and t["targets"][2]}
    systems = {ACTION_TO_SYSTEM[a] for a in acts if a in ACTION_TO_SYSTEM}
    # FAQ lookups and scripted troubleshooting ("log out and back in", "try again") are documentation work.
    if any(a.startswith("search-") or a in ("select-faq", "ask-the-oracle", "log-out-in", "try-again", "instructions") for a in acts):
        systems.add("knowledge_base")
    return dict(flow=convo["scenario"]["flow"], intent=re.sub(r"_\d+$", "", convo["scenario"]["subflow"]),
                systems=sorted(systems), turns=len(convo["original"]))


def extract(convos, cache):
    done = {}
    if cache.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in cache.open()}
    todo = [c for c in convos if str(c["convo_id"]) not in done]

    def one(c):
        prompt = EXTRACT_PROMPT % (list(SYSTEMS), json.dumps(SYSTEMS), transcript(c))
        for _ in range(3):  # cheaper models sometimes drop a field; ask again rather than guess it
            ans, usage = llm(prompt)
            if {"reason", "automatable", "systems"} <= ans.keys() and ans["automatable"] in ("full", "partial", "no"):
                break
        else:
            ans.setdefault("reason", "unclear"); ans["automatable"] = "unknown"; ans.setdefault("systems", [])
        ans.setdefault("why", ""); ans.setdefault("info_needed", [])
        ans["systems"] = [s for s in ans.get("systems", []) if s in SYSTEMS]
        return dict(id=str(c["convo_id"]), pred=ans, usage=usage)

    with ThreadPoolExecutor(8) as pool, cache.open("a") as f:
        for i, row in enumerate(pool.map(one, todo)):
            f.write(json.dumps(row) + "\n"); f.flush()
            done[row["id"]] = row
            if i % 25 == 0:
                print(f"  extracted {len(done)}/{len(convos)}", file=sys.stderr)
    return [done[str(c["convo_id"])] for c in convos]


def taxonomy(rows, cache, max_types=12):
    if cache.exists():
        return json.load(cache.open())
    lines = "\n".join(f"{i}: {r['pred']['reason']}" for i, r in enumerate(rows))
    ans, _ = llm(TAXONOMY_PROMPT % (max_types, lines), max_tokens=16000, timeout=900)
    label = {}
    for t in ans["types"]:
        for i in t["ids"]:
            label[int(i)] = t["name"]
    json.dump(label, cache.open("w"))
    return {str(k): v for k, v in label.items()}


def purity(pred, gold):
    """Share of calls whose discovered type's majority true label matches their own."""
    by = defaultdict(list)
    for p, g in zip(pred, gold):
        by[p].append(g)
    return sum(Counter(v).most_common(1)[0][1] for v in by.values()) / len(gold)


def nmi(pred, gold):
    import math
    n = len(gold); cp, cg, cj = Counter(pred), Counter(gold), Counter(zip(pred, gold))
    mi = sum(c / n * math.log(c * n / (cp[p] * cg[g])) for (p, g), c in cj.items())
    h = lambda C: -sum(c / n * math.log(c / n) for c in C.values())
    return 2 * mi / (h(cp) + h(cg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    # Savings assumptions. They are inputs, not findings; the report prints them next to the result.
    ap.add_argument("--calls-per-month", type=int, default=20000)
    ap.add_argument("--handle-min", type=float, default=6.0, help="average human handle time per call, minutes")
    ap.add_argument("--agent-cost-hr", type=float, default=30.0, help="fully loaded cost of a human agent hour, USD")
    ap.add_argument("--ai-cost-min", type=float, default=0.09, help="voice AI cost per minute, USD")
    ap.add_argument("--partial-share", type=float, default=0.5, help="share of handle time saved on 'partial' calls")
    a = ap.parse_args()

    data = json.load(gzip.open(ABCD))[a.split]
    convos = random.Random(a.seed).sample(data, a.n)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.split}_{a.n}_s{a.seed}_{MODEL.split('/')[-1]}"
    rows = extract(convos, OUT / f"extract_{tag}.jsonl")
    label = taxonomy(rows, OUT / f"taxonomy_{tag}.json")
    gold = [truth(c) for c in convos]
    types = [label.get(str(i), "unassigned") for i in range(len(rows))]

    # How good is it? Scored against labels the model never saw.
    sys_tp = Counter(); sys_fp = Counter(); sys_fn = Counter()
    for r, g in zip(rows, gold):
        p, t = set(r["pred"]["systems"]), set(g["systems"])
        for s in p & t: sys_tp[s] += 1
        for s in p - t: sys_fp[s] += 1
        for s in t - p: sys_fn[s] += 1
    lines = [f"# Transcript audit ({a.split}, {a.n} conversations, model {MODEL})", ""]
    lines += ["## Accuracy against hidden labels", "",
              f"Call types found: {len(set(types))} (true intents in sample: {len(set(g['intent'] for g in gold))}, "
              f"true top-level flows: {len(set(g['flow'] for g in gold))})", "",
              "| Measure | vs 10 flows | vs fine intents |", "|---|---|---|",
              f"| Purity (calls in a type that share its majority label) | {purity(types, [g['flow'] for g in gold]):.0%} | {purity(types, [g['intent'] for g in gold]):.0%} |",
              f"| NMI | {nmi(types, [g['flow'] for g in gold]):.2f} | {nmi(types, [g['intent'] for g in gold]):.2f} |", "",
              "Systems the agent would need (truth = actions the human agent actually took):", "",
              "| System | Precision | Recall | Calls needing it |", "|---|---|---|---|"]
    for s in SYSTEMS:
        tp, fp, fn = sys_tp[s], sys_fp[s], sys_fn[s]
        if tp + fp + fn == 0:
            continue
        pr = f"{tp / (tp + fp):.0%}" if tp + fp else "n/a"; rc = f"{tp / (tp + fn):.0%}" if tp + fn else "n/a"
        lines.append(f"| {s} | {pr} | {rc} | {tp + fn} |")
    tp, fp, fn = sum(sys_tp.values()), sum(sys_fp.values()), sum(sys_fn.values())
    lines.append(f"| **all** | **{tp / (tp + fp):.0%}** | **{tp / (tp + fn):.0%}** | |")

    # The report a customer would get.
    by_type = defaultdict(list)
    for t, r in zip(types, rows):
        by_type[t].append(r["pred"])
    lines += ["", "## What customers call about", "",
              "| Call type | Share | Fully automatable | Partly | Not | Systems needed (most common) |", "|---|---|---|---|---|---|"]
    for t, ps in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        auto = Counter(p["automatable"] for p in ps)
        syst = Counter(s for p in ps for s in p["systems"]).most_common(3)
        lines.append(f"| {t} | {len(ps) / len(rows):.0%} | {auto['full'] / len(ps):.0%} | {auto['partial'] / len(ps):.0%} | "
                     f"{auto['no'] / len(ps):.0%} | {', '.join(s for s, _ in syst)} |")
    # Calibration check: the human agent's own escalations are the only automation ground truth ABCD has.
    esc = Counter(); rest = Counter()
    for r, g in zip(rows, gold):
        (esc if "human_handoff" in g["systems"] else rest)[r["pred"]["automatable"]] += 1
    lines += ["", "## Is the automation judgment calibrated?", "",
              "ABCD has no automation label, but it records when the human agent escalated to another team. "
              "Those calls should rarely be judged fully automatable.", "",
              "| Calls | n | full | partial | no | unknown |", "|---|---|---|---|---|---|"]
    for name, c in (("human agent escalated", esc), ("all other calls", rest)):
        n = sum(c.values())
        lines.append(f"| {name} | {n} | " + " | ".join(f"{c[k] / n:.0%}" for k in ("full", "partial", "no", "unknown")) + " |")

    auto = Counter(r["pred"]["automatable"] for r in rows)
    full, part = auto["full"] / len(rows), auto["partial"] / len(rows)
    hours = a.calls_per_month * a.handle_min / 60
    low, high = hours * full, hours * (full + part)  # partial calls save somewhere between none and all of their time
    lines += ["", "## Savings, as a range", "",
              f"Judged fully automatable {full:.0%}, partly {part:.0%}, not {auto['no'] / len(rows):.0%}"
              + (f", no usable answer {auto['unknown'] / len(rows):.0%}" if auto["unknown"] else "") + ".", "",
              "| Assumption (change with flags) | Value |", "|---|---|",
              f"| Calls per month | {a.calls_per_month:,} |", f"| Human handle time | {a.handle_min} min |",
              f"| Human agent cost | ${a.agent_cost_hr}/hr |", f"| Voice AI cost | ${a.ai_cost_min}/min |", "",
              f"Agent hours freed per month: {low:,.0f} to {high:,.0f} of {hours:,.0f}, "
              f"worth ${low * a.agent_cost_hr:,.0f} to ${high * a.agent_cost_hr:,.0f} before the voice AI cost "
              f"(${a.calls_per_month * full * a.handle_min * a.ai_cost_min:,.0f} to "
              f"${a.calls_per_month * (full + part) * a.handle_min * a.ai_cost_min:,.0f}).", "",
              "The automation split is the model's judgment and runs optimistic (see the table above). "
              "The systems and call-type numbers are what was measured. Use this range to decide what to "
              "pilot, not to quote a saving.", ""]
    examples = [(t, r["pred"]) for t, r in zip(types, rows)]
    lines += ["## Example judgments", ""]
    for t, p in examples[:6]:
        lines.append(f"- **{t}** / {p['reason']}: {p['automatable']}. {p['why']} Systems: {', '.join(p['systems']) or 'none'}.")
    usage = [r["usage"] for r in rows if r.get("usage")]
    lines += ["", f"Tokens: {sum(u.get('prompt_tokens', 0) for u in usage):,} in, {sum(u.get('completion_tokens', 0) for u in usage):,} out. "
              f"API cost for the per-conversation pass: ${sum(u.get('cost', 0) for u in usage):.2f}."]
    report = "\n".join(lines)
    (OUT / f"report_{tag}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
