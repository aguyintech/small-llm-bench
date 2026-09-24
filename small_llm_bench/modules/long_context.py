"""Long-context module: needle retrieval and reasoning over a large haystack.

The haystack (~16k–64k tokens of filler) is built deterministically at runtime
so the wall of text is never committed to YAML. Sizes are approximate: without
a tokenizer dependency the generator counts words, and real tokenization of the
filler lands within roughly 10% of the declared ``filler_tokens``. Task types (RULER / NoLiMa lineage):
  single       — one keyword needle at a given depth.
  no_overlap   — needle shares no keywords with the question (latent association).
  multi_key    — target needle among similar distractor needles.
  multi_hop    — a chain of variable bindings; answer the final value. Optional
                 ``latent`` lines state a link by association rather than by
                 name, and optional ``distractors`` add near-miss bindings.
  aggregation  — a word injected N times; answer the most frequent one.
Graded by reusing the knowledge scorer (numeric / factual). Models whose context
window is too small error out — itself signal (counts as a failure).
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING, Any

from ..models import Task, TaskResult, TurnRecord
from .base import BaseModule, completion_tokens, hit_length_cap, message_text

if TYPE_CHECKING:
    from ..runner import ChatClient

_FILLER = ("Log entry {i}: routine system check completed for node {n} with "
           "nominal status and no anomalies recorded across all monitored subsystems.")


# Average characters per token for ordinary English prose. Used only to
# sanity-check generated haystack sizes — the generator itself counts words.
_CHARS_PER_TOKEN = 4.7


def estimate_tokens(text: str) -> int:
    """Approximate token count of a built haystack, for tests and reporting."""
    return int(len(text) / _CHARS_PER_TOKEN)


def _salt_offset(salt: str) -> int:
    """Starting log-entry number for a salted haystack (0 when unsalted).

    Every haystack used to start at ``Log entry 1`` from the same template, so
    each document was a prefix of every longer one up to its first needle —
    thousands of identical lines for the 32k/48k tasks. A server with prefix
    caching on (llama.cpp and vLLM both default to it) then skipped most of the
    prefill for every long-context task after the first. Scores were unaffected,
    but wall-clock and prefill_seconds were understated, which is exactly the
    number the runtime budget is fitted against. Offsetting the line numbering
    per task makes each document unique from its first line.

    sha256, not ``hash()``: the offset has to be stable across processes.
    """
    if not salt:
        return 0
    return int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16) % 1_000_000


# --- realistic filler -------------------------------------------------------
# The `log` style above is one sentence repeated with two numbers changing.
# Measured 2026-09-07, that makes every needle-based task easy for the wrong
# reason: a needle written in ANY other shape is the only line of its shape in
# the document, so retrieval is shape-matching rather than search. All three
# trio models pulled a 44-character API key out of 54k tokens of it verbatim,
# the 4B included, and the retired lc_45 at 48k was cleared 3/3 the same way.
#
# `ops_audit` is what the document those tasks pretend to be actually looks
# like: mixed event shapes, real-looking identifiers, several credential
# families, named actors, and occasional prose. The identifiers are the point.
# A secrets-audit log is FULL of token-shaped strings, so realistic filler and
# interfering filler are the same thing — a key-shaped needle has nowhere to
# hide in it.
_SERVICES = ("billing-api", "billing-worker", "billing-api-staging",
             "ledger-api", "ledger-worker", "invoice-api", "payments-api",
             "payments-worker", "auth-api", "auth-worker", "search-api",
             "notify-api", "export-worker", "recon-api", "webhook-api",
             "gateway-edge", "session-api", "audit-sink")
_ACTORS = ("m.oyelaran", "s.kowalczyk", "deploybot", "t.nakamura", "a.ferreira",
           "rotation-cron", "j.okonkwo", "terraform-ci", "l.bergstrom",
           "pagerduty-sync", "r.dasgupta", "vault-agent")
_ENVS = ("prod", "prod", "prod", "staging", "canary")
_REGIONS = ("eu-west-1", "eu-north-1", "us-east-1", "ap-northeast-1")
_PATHS = ("/v2/invoices", "/v2/charges", "/v1/customers", "/v2/refunds",
          "/health", "/v2/payouts", "/v1/webhooks", "/v2/disputes")
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_OPS_TEMPLATES = (
    "{ts} {env} {svc} deploy {sha} by {actor}: rollout complete, {n} pods healthy",
    "{ts} {env} {svc} GET {path} 200 {n}ms trace={trace} req={req}",
    "{ts} {env} {svc} POST {path} 201 {n}ms trace={trace} req={req}",
    "{ts} audit actor={actor} action=secret.read scope={svc} ref={req} result=allow",
    "{ts} audit actor={actor} action=secret.list scope={svc} ref={req} result=allow",
    "{ts} {env} {svc} webhook signing secret verified whsec_{k24} attempts=1",
    "{ts} {env} {svc} publishable key in use pk_live_{k24} region={region}",
    "{ts} {env} {svc} test-mode credential issued sk_test_{k36} ttl=3600s",
    "{ts} {env} {svc} scheduled key audit: no findings, checksum {sha}",
    "{ts} {env} {svc} connection pool resized to {n} by {actor}",
    "{ts} {env} {svc} cache eviction pass freed {n}MB, region={region}",
    "{ts} INC-{n} {actor} noted elevated latency on {svc} in {region}; no customer impact,"
    " monitoring continues and the on-call handover has been updated accordingly.",
    "{ts} {env} {svc} config change by {actor}: timeout {n}ms, retries 3, ref={req}",
    "{ts} {env} {svc} certificate renewed, fingerprint {sha}, expiry 2027-01-14",
    "{ts} {env} {svc} vault lease renewed for {actor}, lease_id={trace}",
    "{ts} {env} {svc} backup snapshot {req} written, {n}GB, region={region}",
)


def _ops_audit_lines(target_tokens: int, salt: str = "") -> list[str]:
    """Deterministic mixed-shape operations/audit filler.

    Seeded from ``salt`` alone, so the same task always builds the same
    document and two tasks never share a prefix (prefix caching would
    otherwise skip most of the prefill — see ``_salt_offset``).
    """
    rnd = random.Random(f"ops-audit::{salt}")

    def token(n: int) -> str:
        return "".join(rnd.choice(_ALNUM) for _ in range(n))

    lines: list[str] = []
    approx = 0
    while approx < target_tokens:
        ts = (f"2026-0{rnd.randrange(3, 9)}-{rnd.randrange(1, 29):02d}T"
              f"{rnd.randrange(0, 24):02d}:{rnd.randrange(0, 60):02d}:"
              f"{rnd.randrange(0, 60):02d}Z")
        line = rnd.choice(_OPS_TEMPLATES).format(
            ts=ts, env=rnd.choice(_ENVS), svc=rnd.choice(_SERVICES),
            actor=rnd.choice(_ACTORS), region=rnd.choice(_REGIONS),
            path=rnd.choice(_PATHS), n=rnd.randrange(2, 9000),
            sha=token(12), trace=token(16), req="req_" + token(20),
            k24=token(24), k36=token(36))
        lines.append(line)
        # Characters, not words*1.3. This filler is dense with opaque
        # identifiers, and a 44-character token is ONE word but about fifteen
        # tokens — the word-based estimate undercounts it by ~4.7x, which made
        # `filler_tokens` meaningless for this style (a declared 4000 measured
        # as ~19000 against the real tokenizer). Measured ratio on the trio is
        # 1.97 chars/token; 2.0 is close enough and errs toward a slightly
        # smaller document, which is the safe direction against a 65,536-token
        # ceiling.
        approx += int(len(line) / 2.0)
    return lines


def _OPS_CHARS_PER_TOKEN_NOTE() -> None:  # pragma: no cover - doc anchor
    """See the estimate inside ``_ops_audit_lines``; kept findable by grep."""


# --- meeting-transcript filler ---------------------------------------------
# For tasks whose answer is a CONCLUSION rather than a string. Every
# long-context construct measured against the 4B floor on 2026-09-07/08 was
# lexical — the answer was a key or a number literally present in the document
# — and the 4B passed all five 3/3, because locating a record is what it is
# good at. A transcript asks something different: no line states the answer, so
# there is no record to locate and no shape or keyword to search on.
#
# The interference here is other people committing to other things. A
# transcript with exactly one ownership resolution in it would be answerable
# by finding "the commitment"; these blocks put several unrelated ones in the
# way, so the question "who owns X" cannot be reduced to "who volunteered".
#
# Nothing in this filler may read as a commitment to a FORECAST or a REBUILD —
# a task grading that thread has to own those words exclusively, or the
# document becomes genuinely ambiguous and the task is broken rather than hard.
_CAST = ("Priya", "Marco", "Dana", "Ines", "Rafael", "Nadia", "Bo", "Tomas")
_TOPICS = (
    ("the on-call rota", "rota"),
    ("the vendor invoice backlog", "invoices"),
    ("staging flakiness", "staging"),
    ("the onboarding docs", "docs"),
    ("the hiring loop", "hiring"),
    ("the dashboard cleanup", "dashboards"),
    ("the log retention change", "retention"),
    ("the API deprecation notice", "deprecation"),
    ("the incident review write-up", "the write-up"),
    ("the storage cost spike", "storage costs"),
)
_CHATTER = (
    "{a}: On {topic} — where did we land last week?",
    "{a}: I looked at {topic} briefly and it is smaller than it sounded.",
    "{b}: Agreed, though the timing is awkward with the release freeze.",
    "{a}: Do we have numbers on that? I saw about {n} last time I checked.",
    "{b}: Roughly {n}, yes, but it moves around week to week.",
    "{a}: Can we park {topic} until after the freeze lifts?",
    "{b}: Fine by me. Nothing there is urgent.",
    "{a}: One caveat — the {topic} work touches the same config as {other}.",
    "{b}: Then we sequence them. {other} first, it is further along.",
    "{a}: I will add a note to the doc so nobody trips over it.",
    "{b}: Sorry, I dropped off for a second. What was the last thing?",
    "{a}: Just that {topic} is parked until the freeze lifts.",
    "{b}: Understood. Anything blocking on my side?",
    "{a}: Not that I know of. It is mostly waiting on review.",
    "{b}: The review queue is about {n} deep, so give it a few days.",
    "{a}: That is fine. It is not on the critical path.",
    "{b}: While we are here — did anyone hear back about {other}?",
    "{a}: Not yet. I chased it on Tuesday and again this morning.",
    "{b}: I would give it until Friday and then escalate.",
    "{a}: Works for me. Let us move on.",
)
# Decoy ownership: other work, clearly resolved, so "who volunteered" is not a
# shortcut to the answer.
_DECOY_OWNERSHIP = (
    "{a}: I can own {topic} — it overlaps what I am already doing.\n"
    "{b}: Great, {a} has {topic} then.",
    "{a}: Does anyone want {topic}?\n"
    "{b}: I will take it.\n"
    "{a}: Thanks {b}, noted as yours.",
    "{a}: {topic} needs an owner before we close.\n"
    "{b}: Put me down, I have the context already.\n"
    "{a}: Done.",
)


def _meeting_lines(target_tokens: int, salt: str = "") -> list[str]:
    """Deterministic multi-speaker meeting filler, with decoy ownership threads.

    Seeded from ``salt`` alone, like ``_ops_audit_lines``, so a task always
    builds the same transcript and two tasks never share a prefix.
    """
    rnd = random.Random(f"meeting::{salt}")
    lines: list[str] = []
    approx = 0
    while approx < target_tokens:
        topic, short = rnd.choice(_TOPICS)
        other = rnd.choice([t[0] for t in _TOPICS if t[0] != topic])
        a, b = rnd.sample(_CAST, 2)
        pool = (_DECOY_OWNERSHIP if rnd.random() < 0.10 else _CHATTER)
        block = rnd.choice(pool).format(a=a, b=b, topic=short,
                                        other=other, n=rnd.randrange(3, 400))
        for line in block.split("\n"):
            lines.append(line)
            approx += int(len(line.split()) * 1.3)
    return lines


def _filler_lines(target_tokens: int, salt: str = "",
                  style: str = "log") -> list[str]:
    """Generate deterministic filler lines up to ~target_tokens (words*1.3).

    Approximate by construction — see ``estimate_tokens`` for the independent
    character-based check. The two agree to within ~10% on this filler.

    ``salt`` shifts the line numbering so two haystacks never share a prefix
    (see ``_salt_offset``); it does not change the line count.

    ``style`` selects the filler shape. ``log`` is the original one-sentence
    template and stays the default so no banked task's content hash moves;
    ``ops_audit`` is the realistic mixed-shape one — see the comment above it
    for why the difference decides whether a needle-based task measures search
    or shape-matching.
    """
    if style == "ops_audit":
        return _ops_audit_lines(target_tokens, salt)
    if style == "meeting":
        return _meeting_lines(target_tokens, salt)
    lines: list[str] = []
    approx = 0
    index = _salt_offset(salt)
    while approx < target_tokens:
        index += 1
        line = _FILLER.format(i=index, n=(index * 7) % 1000)
        lines.append(line)
        approx += int(len(line.split()) * 1.3)
    return lines


def _insert_at(lines: list[str], item: str, position: float) -> None:
    """Insert item at a fractional depth (0=start, 1=end)."""
    idx = min(len(lines), max(0, int(len(lines) * position)))
    lines.insert(idx, item)


def build_haystack(spec: dict[str, Any], salt: str = "") -> str:
    """Build a deterministic haystack document for the given spec.

    Common keys: ``filler_tokens``, ``filler_style`` (default 'log', or
    'ops_audit' for realistic mixed-shape filler), ``type`` (default
    'single'). Type-specific:
    single/no_overlap → ``needle`` (+ ``position``); multi_key → ``needle`` +
    ``distractors``; multi_hop → ``chain`` (list of binding lines, plus optional
    ``latent`` and ``distractors``); aggregation → ``inject`` ({word: count}).

    ``salt`` (the task id, in a real run) keeps two haystacks from sharing a
    filler prefix — see ``_salt_offset``. Same spec plus same salt always builds
    the same document.
    """
    kind = spec.get("type", "single")
    lines = _filler_lines(spec.get("filler_tokens", 8000), salt,
                          style=spec.get("filler_style", "log"))

    if kind == "multi_hop":
        chain = spec["chain"]
        for j, line in enumerate(chain):
            pos = 0.1 + 0.8 * (j / max(1, len(chain) - 1)) if len(chain) > 1 else 0.5
            _insert_at(lines, line, pos)
        # A chain whose links all name their variable measures bookkeeping over
        # distance. `latent` states one link by association instead, so the
        # model has to make a NoLiMa-style world-knowledge hop *and* hold the
        # chain — the two constructs that separate models best, in one task.
        for j, line in enumerate(spec.get("latent", [])):
            _insert_at(lines, line, 0.2 + 0.6 * ((j + 1) / (len(spec["latent"]) + 1)))
        for j, line in enumerate(spec.get("distractors", [])):
            _insert_at(lines, line, 0.15 + 0.6 * (j / max(1, len(spec["distractors"]))))
        return "\n".join(lines)

    if kind == "aggregation":
        items: list[str] = []
        for word, count in spec["inject"].items():
            items += [f"Note: the keyword {word} is mentioned here."] * count
        # Interleave before placing. The positions below increase
        # monotonically, so inserting the flattened list in declaration order
        # put every mention of a word in one contiguous block — which turns
        # "which appears most often" into "which block is longest", answerable
        # from one screen of a 24k document. No task reached this branch before
        # v0.13, so the defect had never shown up. Seeded from the spec itself,
        # so the same spec always builds the same document.
        random.Random(repr(sorted(spec["inject"].items()))).shuffle(items)
        for j, item in enumerate(items):
            _insert_at(lines, item, (j + 1) / (len(items) + 1))
        return "\n".join(lines)

    # needle-based: single / no_overlap / multi_key
    distractors = list(spec.get("distractors", []))
    for j, nd in enumerate(distractors):
        _insert_at(lines, nd, 0.15 + 0.6 * (j / max(1, len(distractors))))
    _insert_at(lines, spec["needle"], spec.get("position", 0.5))
    return "\n".join(lines)


class LongContextModule(BaseModule):
    """Tests retrieval and reasoning over a large context window."""

    name = "long_context"

    async def run_task(self, client: "ChatClient", task: Task,
                       sandbox: dict[str, Any] | None = None) -> TaskResult:
        """Build the haystack, ask the question, and record the answer."""
        document = build_haystack(task.haystack, salt=task.id)
        content = f"{document}\n\n{task.prompt}"
        response = await client.chat([{"role": "user", "content": content}],
                                     max_tokens=task.max_tokens)
        message = response["choices"][0]["message"]
        answer = message_text(message)
        return TaskResult(
            task_id=task.id,
            module=self.name,
            prompt=task.prompt,
            expected={**task.expected, "answer_type": task.answer_type},
            response_raw=answer,
            turns=[TurnRecord(role="assistant", content=answer,
                              completion_tokens=completion_tokens(response),
                              truncated=hit_length_cap(response))],
            completion_tokens=completion_tokens(response),
            truncated=hit_length_cap(response),
            raw_api_responses=[response] if client.save_responses else None,
        )
