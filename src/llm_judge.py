#!/usr/bin/env python3
"""Kuku Stage 4 — Contextual LLM judge orchestrator.

Consumes the retrieval union (`retrieval_output/fast_discovery_results.json`),
re-attaches each candidate's FULL context-builder payload
(`context_output/candidates.json`) and asks an LLM to judge promo suitability
blindly: no category matches, scores or ranks ever enter the prompt.

Provider contract (`LlmJudgeProvider`): anything exposing
`complete_messages(messages: list[dict]) -> LlmCompletion`. Shipped providers:

* ``LlamaServerProvider`` — the llama.cpp OpenAI-compatible endpoint used for
  staging (chat_template_kwargs disables the coder model's reasoning preamble
  so the content channel carries plain JSON).
* ``OfflineScriptedJudge`` — deterministic token-overlap double used by the
  test suite (``--provider offline``); never production.

Re-runs reuse a content-addressed JSONL cache (`--no-reuse` to force), keyed by
prompt version + provider + what the model actually sees, so anonymous cheaper
re-runs don't re-score everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import judge_prompt
import judge_schema as js

OUTPUT_SCHEMA_VERSION = "kuku-judge-output-1.0.0"
RETRIEVAL_RESULTS_NAME = "fast_discovery_results.json"
RESULTS_JSON = "judge_results.json"
RESULTS_JSONL = "judge_results.jsonl"
REPORT_JSON = "judge_report.json"
CACHE_JSONL = "judge_cache.jsonl"

DEFAULT_BASE_URL = "http://100.90.211.24:9931"
DEFAULT_MODEL = "C:\\llama-cpp\\Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf"


@dataclass
class LlmCompletion:
    """One raw exchange with a provider."""

    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    raw: Optional[dict] = None


@runtime_checkable
class LlmJudgeProvider(Protocol):
    """Anything that can chat with full messages. The judge never sees providers behind this seam."""

    model_id: str

    def complete_messages(self, messages: list[dict], *, max_output_tokens: int, temperature: float) -> LlmCompletion: ...


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class LlamaServerProvider:
    """llama.cpp OpenAI-compatible chat completions endpoint.

    Notes for this staging host (probed 2026-11-08):
    * Model emits its scratchpad in ``reasoning_content`` when templates allow
      thinking; we pass ``chat_template_kwargs={"enable_thinking": false}`` so
      the content channel carries the contracted JSON directly.
    * Usage blocks are returned by the server (prompt/completion tokens and
      timings) and are recorded for the runtime/token report.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 600.0,
        transport_retries: int = 2,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model
        self.timeout_seconds = timeout_seconds
        self.transport_retries = transport_retries
        self.disable_thinking = disable_thinking

    def complete_messages(self, messages: list[dict], *, max_output_tokens: int, temperature: float) -> LlmCompletion:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Optional[Exception] = None
        backoff = 2.0
        for attempt in range(self.transport_retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return self._parse_response(raw, time.perf_counter() - started)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < self.transport_retries:
                    time.sleep(backoff)
                    backoff *= 2
        raise RuntimeError(f"Provider transport failed after {self.transport_retries + 1} attempts: {last_error}")

    @staticmethod
    def _parse_response(raw: dict, latency: float) -> LlmCompletion:
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = raw.get("usage") or {}
        return LlmCompletion(
            text=str(message.get("content") or ""),
            reasoning=str(message.get("reasoning_content") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_seconds=latency,
            raw=raw,
        )


class OfflineScriptedJudge:
    """Deterministic double for tests/headless runs. Arbitrary but valid JSON.

    Scores derive from simple text features of the payload (fire any of the
    configured tokens to push a dimension up); verdict accepts above weighted
    3.0, which leaves both outcomes exercised in tests.
    """

    def __init__(self, mode: str = "heuristic"):
        if mode not in ("heuristic", "accept_all", "reject_all"):
            raise ValueError("mode must be heuristic|accept_all|reject_all")
        self.mode = mode
        self.model_id = f"offline-scripted-judge-v1-{mode}"

    _HOT = ("betray", "died", "killed", "abandon", "stolen", "witch", "traitor", "say goodbye")
    _QUESTION_SUFFIXES = ("?", "...", "…")

    def complete_messages(self, messages: list[dict], *, max_output_tokens: int, temperature: float) -> LlmCompletion:
        payload_text = next(
            (message.get("content", "") for message in messages if message.get("role") == "user"),
            "",
        )
        body = payload_text[payload_text.find("{"):] if "{" in payload_text else payload_text
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        dialogue = payload.get("dialogue") or []
        first3 = " ".join(str(item.get("content", "")).lower() for item in dialogue[:3])
        joined = " ".join(str(item.get("content", "")).lower() for item in dialogue)
        deliveries = " ".join(
            str(item.get("delivery", "") or "") for item in dialogue
        ).lower()
        last_line = str((dialogue or [{}])[-1].get("content", "")).strip().lower() if dialogue else ""
        ids = [item.get("event_id") for item in dialogue if item.get("event_id")] or ["<unknown>"]

        if self.mode == "accept_all":
            opening, cliff = 5, 4
        elif self.mode == "reject_all":
            opening, cliff = 1, 1
        else:
            opening = 4 if any(token in first3 for token in self._HOT) else 2
            cliff = 4 if any(last_line.endswith(sfx) for sfx in self._QUESTION_SUFFIXES) else 2
        standalone = 3 + (1 if payload.get("episode_logline") else 0)
        progression = 4 if len(dialogue) >= 3 else 2
        impact = 4 if any(w in deliveries for w in ("trembling", "furious", "defiant", "shocked", "broken", "tormented")) else 2
        scores = {
            "opening_hook": opening,
            "cliffhanger": cliff,
            "standalone_clarity": standalone,
            "narrative_progression": progression,
            "emotional_impact": impact,
        }
        weighted = js.weighted_score(scores)
        accept = weighted >= 3.0 if self.mode == "heuristic" else (self.mode == "accept_all")
        evidence = [
            {"dimension": dimension, "statement": f"offline heuristic for {dimension}", "event_ids": [ids[i % len(ids)]]}
            for i, dimension in enumerate(js.DIMENSIONS)
        ]
        answer = {
            "scores": scores,
            "verdict": js.VERDICT_ACCEPT if accept else js.VERDICT_REJECT,
            "confidence": 0.72,
            "observed_arc": "altar_abandonment",
            "evidence": evidence,
            "rejection_reasons": [] if accept else ["weak_opening"],
            "boundary_feedback": None,
            "notes": "Synthetic offline judgement for pipeline testing, not for production.",
        }
        text = json.dumps(answer, ensure_ascii=False)
        prompt_tokens = max(1, len(payload_text.split()))
        return LlmCompletion(
            text=text,
            reasoning="",
            prompt_tokens=prompt_tokens,
            completion_tokens=max(1, len(text.split())),
            latency_seconds=0.001,
            raw={"provider": self.model_id},
        )


def build_provider(name: str, args: argparse.Namespace) -> LlmJudgeProvider:
    if name == "offline":
        return OfflineScriptedJudge(mode=getattr(args, "offline_mode", "heuristic"))
    if name == "llama-server":
        return LlamaServerProvider(
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=args.timeout,
        )
    raise ValueError(f"Unknown provider {name!r}; valid: llama-server|offline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_output_tokens: int = 1200
    repair_attempts: int = 2
    include_retrieval_metadata: bool = False
    reuse_cache: bool = True
    limit_candidates: Optional[int] = None


# ---------------------------------------------------------------------------
# JSON extraction + validation with repair
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> dict:
    """Parse the FIRST balanced {...} object in `text`, fences tolerated."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        body = stripped[first_newline + 1:] if first_newline != -1 else stripped
        closing = body.rfind("```")
        stripped = body[:closing] if closing != -1 else body
    start = stripped.find("{")
    if start == -1:
        raise ValueError("model returned no JSON object.")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(stripped[start:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object.")
    return obj


def _messages_from_attempt(messages: list[dict], completion_text: str, errors: list[str]) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": completion_text},
        {"role": "user", "content": (
            "Your reply failed validation. Problems:\n- "
            + "\n- ".join(errors)
            + "\nReturn corrected JSON only — no commentary, no code fences. All other content must be removed."
        )},
    ]


def judge_one(
    candidate: dict,
    provider: LlmJudgeProvider,
    config: JudgeConfig,
) -> dict:
    """Judge a single candidate with repairs; returns the result row."""
    candidate_id = candidate.get("candidate_id")
    system = judge_prompt.SYSTEM_PROMPT
    user = judge_prompt.build_user_message(
        candidate, include_retrieval_metadata=config.include_retrieval_metadata
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    attempts: list[dict] = []
    total_prompt = total_completion = 0
    total_latency = 0.0
    final_judgement: Optional[dict] = None
    final_raw_text = ""
    final_errors: list[str] = []

    for attempt in range(config.repair_attempts + 1):
        try:
            completion = provider.complete_messages(
                messages,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
            )
        except RuntimeError as exc:
            attempts.append({"attempt": attempt + 1, "ok": False, "errors": [str(exc)],
                             "prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": 0.0})
            final_errors = [str(exc)]
            break

        total_prompt += completion.prompt_tokens
        total_completion += completion.completion_tokens
        total_latency += completion.latency_seconds
        raw_text = completion.text
        final_raw_text = raw_text
        try:
            decoded = extract_json_object(raw_text)
            judgement = js.validate_judgement(decoded, candidate)
            final_judgement = js.attach_weighted_score(judgement)
            attempts.append({
                "attempt": attempt + 1, "ok": True, "errors": [],
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "latency_seconds": round(completion.latency_seconds, 3),
            })
            break
        except js.JudgementValidationError as exc:
            errors = exc.errors
        except ValueError as exc:
            errors = [str(exc)]
        attempts.append({
            "attempt": attempt + 1, "ok": False, "errors": errors,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "latency_seconds": round(completion.latency_seconds, 3),
        })
        final_errors = errors
        if attempt < config.repair_attempts:
            messages = _messages_from_attempt(messages, raw_text, errors)

    row = {
        "candidate_id": candidate_id,
        "status": "ok" if final_judgement is not None else "failed",
        "fingerprint": js.fingerprint_for_cache(candidate),
        "cache_hit": False,
        "judgement": final_judgement,
        "attempts": attempts,
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "latency_seconds": round(total_latency, 3),
        },
        **({"error": "; ".join(final_errors)} if final_judgement is None else {}),
        "raw_model_output": final_raw_text,
    }
    return row


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_candidates(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
    else:
        candidates = payload
    if not isinstance(candidates, list):
        raise ValueError(f"No candidates list in {path}")
    return candidates


def load_retrieval_union_ids(retrieval_results: Path) -> list[str]:
    payload = json.loads(retrieval_results.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"No results list in {retrieval_results}")
    ids = []
    for row in rows:
        cid = row.get("candidate_id")
        if cid:
            ids.append(cid)
    return ids


def retrieval_similarity_lookup(retrieval_results: Path) -> dict[str, dict[str, float]]:
    """candidate_id -> full similarity row (only for the evaluation comparison)."""
    payload = json.loads(retrieval_results.read_text(encoding="utf-8"))
    return {
        row["candidate_id"]: (row.get("similarity_scores_all_categories") or {})
        for row in payload.get("results", [])
        if row.get("candidate_id")
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(provider: LlmJudgeProvider, candidate: dict, config: JudgeConfig) -> str:
    digest = hashlib.sha1()
    for part in (
        provider.model_id,
        judge_prompt.JUDGE_PROMPT_VERSION,
        str(config.include_retrieval_metadata),
        js.fingerprint_for_cache(candidate),
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("cache_key")
            if key:
                cache[key] = record
    return cache


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_judge(
    candidates: list[dict],
    provider: LlmJudgeProvider,
    config: JudgeConfig,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / CACHE_JSONL
    cache = load_cache(cache_path) if config.reuse_cache else {}

    rows: list[dict] = []
    totals = {"judged_now": 0, "cache_hits": 0, "failed": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": 0.0}

    selected = candidates if config.limit_candidates is None else candidates[: config.limit_candidates]
    for index, candidate in enumerate(selected, start=1):
        candidate_id = candidate.get("candidate_id")
        cache_key = _cache_key(provider, candidate, config)
        cached = cache.get(cache_key)
        if cached is not None and cached.get("judgement") is not None:
            row = json.loads(json.dumps(cached))  # deep copy of cache entry
            row.pop("cache_key", None)
            row["cache_hit"] = True
            totals["cache_hits"] += 1
            print(f"[{index}/{len(selected)}] {candidate_id}: cache hit | verdict={row['judgement']['verdict']} score={row['judgement']['weighted_score']}", flush=True)
            rows.append(row)
            continue

        row = judge_one(candidate, provider, config)
        usage = row["usage"]
        totals["judged_now"] += 1
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]
        totals["latency_seconds"] += usage["latency_seconds"]
        if row["status"] != "ok":
            totals["failed"] += 1
            print(f"[{index}/{len(selected)}] {candidate_id}: FAILED after {len(row['attempts'])} attempts | {row.get('error', '')[:120]}", flush=True)
        else:
            verdict = row["judgement"]["verdict"]
            score = row["judgement"]["weighted_score"]
            print(f"[{index}/{len(selected)}] {candidate_id}: {verdict} score={score:.3f} latency={usage['latency_seconds']:.1f}s", flush=True)
        # Crash-safe: persist the cache line immediately so a killed run keeps all progress.
        record = {**row, "cache_key": cache_key}
        cache[cache_key] = record
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows.append(row)

    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "engine": "llm_judge.py",
        "prompt_version": judge_prompt.JUDGE_PROMPT_VERSION,
        "provider": provider.model_id,
        "config": {
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "repair_attempts": config.repair_attempts,
            "include_retrieval_metadata": config.include_retrieval_metadata,
            "blind": not config.include_retrieval_metadata,
        },
        "totals": {
            **totals,
            "latency_seconds": round(totals["latency_seconds"], 3),
            "candidates_in_run": len(rows),
        },
        "results": rows,
    }
    _write_json_atomic(output_dir / RESULTS_JSON, payload)
    _write_jsonl(output_dir / RESULTS_JSONL, rows)
    _write_json_atomic(output_dir / REPORT_JSON, {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "engine": "llm_judge.py",
        "prompt_version": judge_prompt.JUDGE_PROMPT_VERSION,
        "provider": provider.model_id,
        "config": payload["config"],
        "totals": payload["totals"],
        "status_counts": _count_by(rows, lambda row: row["status"]),
        "verdict_counts": _count_by(
            [row for row in rows if row["status"] == "ok"],
            lambda row: row["judgement"]["verdict"],
        ),
        "weighted_score_mean": _safe_mean(
            [row["judgement"]["weighted_score"] for row in rows if row["status"] == "ok"]
        ),
    })
    return payload


def _count_by(rows: list[dict], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = key_fn(row)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _safe_mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Contextual LLM judge over retrieved promo candidates.")
    parser.add_argument("--retrieval-results", default="retrieval_output/fast_discovery_results.json")
    parser.add_argument("--candidates", default="context_output/candidates.json")
    parser.add_argument("--output", default="judge_output")
    parser.add_argument("--provider", default="llama-server", choices=["llama-server", "offline"])
    parser.add_argument("--offline", action="store_true", help="Alias for --provider offline")
    parser.add_argument("--offline-mode", default="heuristic", choices=["heuristic", "accept_all", "reject_all"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--no-reuse", action="store_true", help="Ignore the judge cache and re-score")
    parser.add_argument("--include-retrieval-metadata", action="store_true",
                        help="NON-BLIND experiment: include retrieval matches in the prompt (recorded in outputs)")
    args = parser.parse_args(argv)

    retrieval_path = Path(args.retrieval_results)
    candidates_path = Path(args.candidates)
    output_dir = Path(args.output)
    try:
        union_ids = load_retrieval_union_ids(retrieval_path)
        candidates_map = {c["candidate_id"]: c for c in load_candidates(candidates_path)}
        missing = [cid for cid in union_ids if cid not in candidates_map]
        if missing:
            raise ValueError(f"{len(missing)} retrieval-union candidates missing from corpus: {missing[:3]}")
        candidates = [candidates_map[cid] for cid in union_ids]
        provider_name = "offline" if args.offline else args.provider
        provider = build_provider(provider_name, args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    config = JudgeConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        include_retrieval_metadata=args.include_retrieval_metadata,
        reuse_cache=not args.no_reuse,
        limit_candidates=args.limit_candidates,
    )
    payload = run_judge(candidates, provider, config, output_dir)
    totals = payload["totals"]
    print("\n=== Judge pass complete ===")
    print(f"Judged now: {totals['judged_now']} | cache hits: {totals['cache_hits']} | failed: {totals['failed']}")
    print(f"Prompt tokens: {totals['prompt_tokens']} | completion tokens: {totals['completion_tokens']}")
    print(f"Latency (accumulated): {totals['latency_seconds']:.1f}s")
    print(f"Results: {output_dir / RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
