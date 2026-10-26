"""x102 stage 3: one API arm judges the sample under the benchmark's verbatim prompt.

OpenRouter transport on the x81 contract (retry/backoff, cost meter with one
cap per arm, resume by row key, append-only JSONL).  Every row stores two
readings of the same reply: the benchmark's own parser (fail-open) and the
x87 parser (fail-closed, with x87's retry conversation on a parse failure).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from x102_common import (
    API_ARMS,
    API_CALLS_PER_ARM,
    API_MAX_TOKENS,
    API_PASSES,
    API_RETRY_MAX_TOKENS,
    API_SEED,
    API_THREADS,
    OPENROUTER_URL,
    PILOT_MIN_PARSE_RATE,
    RETRY_INSTRUCTION,
    XSTEST_PATH,
    benchmark_parse,
    judgments_path,
    load_manifest,
    load_sample,
    read_json,
    read_jsonl,
    render_benchmark_prompt,
    require_registered,
    update_manifest,
    x87_parse,
)
from x81_judge_refusal import load_api_key


class CostMeter:
    """x81's meter with this arm's own prices as the fallback when `cost` is absent."""

    def __init__(self, cap: float, usd_per_m_in: float, usd_per_m_out: float):
        self.cap = float(cap)
        self.usd_per_m_in = usd_per_m_in
        self.usd_per_m_out = usd_per_m_out
        self.lock = threading.Lock()
        self.usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.aborted = False

    def add(self, usage: dict) -> float:
        cost = usage.get("cost")
        if cost is None:
            cost = (
                usage.get("prompt_tokens", 0) * self.usd_per_m_in
                + usage.get("completion_tokens", 0) * self.usd_per_m_out
            ) / 1e6
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        with self.lock:
            self.usd += float(cost)
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
            self.reasoning_tokens += int(reasoning)
            self.calls += 1
            if self.usd > self.cap:
                self.aborted = True
        return float(cost)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "usd": self.usd,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "calls": self.calls,
            }


def call(session: requests.Session, key: str, body: dict) -> dict:
    for backoff in (1, 4, 15, 60, 120):
        try:
            response = session.post(
                OPENROUTER_URL, json=body, timeout=180, headers={"Authorization": f"Bearer {key}"}
            )
        except requests.RequestException:
            time.sleep(backoff)
            continue
        if response.status_code == 200:
            try:
                parsed = response.json()
            except (ValueError, json.JSONDecodeError):
                time.sleep(backoff)
                continue
            if "error" in parsed:
                time.sleep(backoff)
                continue
            return parsed
        if response.status_code in (408, 429, 500, 502, 503, 524):
            time.sleep(backoff)
            continue
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    raise RuntimeError("retries exhausted")


def request_body(spec: dict, messages: list[dict], max_tokens: int, reasoning_effort: str) -> dict:
    body = {
        "model": spec["slug"],
        "messages": messages,
        "max_tokens": max_tokens,
        "seed": API_SEED,
        "usage": {"include": True},
        "reasoning": {"effort": reasoning_effort},
    }
    if spec["temperature"] is not None:
        body["temperature"] = spec["temperature"]
    return body


def usage_record(usage: dict, cost: float) -> dict:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "cost_usd": cost,
    }


def judge_row(
    key: str, row: dict, arm: str, spec: dict, pass_name: str, reasoning_effort: str, meter: CostMeter
) -> dict | None:
    if meter.aborted:
        return None
    prompt = render_benchmark_prompt(row["prompt"], row["response"])
    base = {
        "key": row["key"],
        "arm": arm,
        "pass": pass_name,
        "model": spec["slug"],
        "request": {
            "temperature": spec["temperature"],
            "seed": API_SEED,
            "max_tokens": API_MAX_TOKENS,
            "reasoning_effort": reasoning_effort,
        },
    }
    session = requests.Session()
    messages = [{"role": "user", "content": prompt}]
    reasons: list[str] = []
    content = None
    usage = {}
    finish = None
    for _attempt in range(3):
        try:
            response = call(session, key, request_body(spec, messages, API_MAX_TOKENS, reasoning_effort))
        except RuntimeError as error:
            reasons.append(f"transport: {error}"[:200])
            continue
        cost = meter.add(response.get("usage", {}))
        usage = usage_record(response.get("usage", {}), cost)
        try:
            choice = response["choices"][0]
            content = choice["message"].get("content")
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as error:
            reasons.append(f"schema: {error}"[:200])
            content = None
            continue
        if content:
            break
        reasons.append(f"empty_content: finish={finish}")
        content = None
    if content is None:
        if finish == "length" and all(r.startswith("empty_content") for r in reasons):
            # The judge spent its whole budget without emitting a verdict (GLM at
            # `minimal` reasons for all 512 tokens on a few rows, deterministically):
            # that is an unparsed verdict, not a transport failure.
            return base | {
                "content": "",
                "finish_reason": finish,
                "usage": usage,
                "label_benchmark": None,
                "benchmark_parse": "no_bracket",
                "label_x87": None,
                "x87_parse": "unparsed",
                "unparsed": True,
                "failed": False,
                "fail_reasons": reasons,
            }
        return base | {"failed": True, "fail_reasons": reasons, "usage": usage}
    label_benchmark, benchmark_mode = benchmark_parse(content)
    label_x87, x87_mode = x87_parse(content)
    record = base | {
        "content": content,
        "finish_reason": finish,
        "usage": usage,
        "label_benchmark": label_benchmark,
        "benchmark_parse": benchmark_mode,
        "label_x87": label_x87,
        "x87_parse": x87_mode,
        "unparsed": False,
        "failed": False,
    }
    if label_x87 is None:
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": RETRY_INSTRUCTION},
        ]
        try:
            response = call(
                session, key, request_body(spec, retry_messages, API_RETRY_MAX_TOKENS, reasoning_effort)
            )
            cost = meter.add(response.get("usage", {}))
            retry_usage = usage_record(response.get("usage", {}), cost)
            retry_content = response["choices"][0]["message"].get("content") or ""
        except (RuntimeError, KeyError, IndexError, TypeError) as error:
            retry_usage = {}
            retry_content = ""
            reasons.append(f"retry_transport: {error}"[:200])
        label_x87, x87_mode = x87_parse(retry_content)
        record.update(
            {
                "retry_content": retry_content,
                "retry_usage": retry_usage,
                "label_x87": label_x87,
                "x87_parse": (f"{x87_mode}_retry" if label_x87 is not None else "unparsed"),
                "unparsed": label_x87 is None,
            }
        )
    if reasons:
        record["fail_reasons"] = reasons
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(API_ARMS))
    parser.add_argument("--pass", dest="pass_name", required=True, choices=API_PASSES)
    parser.add_argument("--cap", type=float, help="USD cap for this invocation (default: the arm's)")
    parser.add_argument("--threads", type=int, default=API_THREADS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reasoning-effort", help="override the arm's registered setting")
    return parser.parse_args()


def select_rows(pass_name: str, manifest: dict) -> list[dict]:
    if pass_name == "xstest":
        xstest = read_json(XSTEST_PATH)
        require_registered(manifest, "xstest_sha256", xstest["xstest_sha256"])
        return xstest["rows"]
    sample = load_sample()
    require_registered(manifest, "sample_sha256", sample["sample_sha256"])
    rows = sample["rows"]
    if pass_name == "main":
        return rows
    keys = set(sample["subsets"][pass_name]["keys"])
    return [row for row in rows if row["key"] in keys]


def main() -> None:
    args = parse_args()
    spec = API_ARMS[args.arm]
    manifest = load_manifest()
    settings = manifest.get("api_arms", {}).get(args.arm, {})
    reasoning_effort = args.reasoning_effort or settings.get("reasoning_effort") or spec["reasoning_effort"]
    if args.pass_name != "pilot" and not settings.get("pilot_passed"):
        raise SystemExit(f"FATAL: arm {args.arm} has no passed pilot recorded in the manifest")
    if args.pass_name != "pilot" and reasoning_effort != settings.get("reasoning_effort"):
        raise SystemExit("FATAL: reasoning setting differs from the one the pilot fixed")
    rows = select_rows(args.pass_name, manifest)
    if args.limit:
        rows = rows[: args.limit]
    out_path = judgments_path(args.arm, args.pass_name)
    done = set()
    for record in read_jsonl(out_path):
        if record.get("failed") or record.get("arm") != args.arm or record.get("pass") != args.pass_name:
            continue
        if record["key"] in done:
            raise SystemExit(f"FATAL: duplicate judged row {record['key']} in {out_path}")
        done.add(record["key"])
    todo = [row for row in rows if row["key"] not in done]
    print(f"x102 {args.arm}/{args.pass_name}: {len(done)} done, {len(todo)} to judge, effort={reasoning_effort}")
    if not todo:
        return
    key = load_api_key()
    meter = CostMeter(args.cap if args.cap is not None else spec["cap_usd"], spec["usd_per_m_in"], spec["usd_per_m_out"])
    counts = {"ok": 0, "unparsed": 0, "failed": 0, "x87_direct": 0, "benchmark_bracket": 0, "reasoning_rows": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = [
            pool.submit(judge_row, key, row, args.arm, spec, args.pass_name, reasoning_effort, meter)
            for row in todo
        ]
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                counts["failed"] += 1
            else:
                counts["ok"] += 1
                counts["unparsed"] += int(record["unparsed"])
                counts["x87_direct"] += int(record["x87_parse"] in ("bracketed", "bare_final_line"))
                counts["benchmark_bracket"] += int(record["benchmark_parse"] != "no_bracket")
                counts["reasoning_rows"] += int(record["usage"]["reasoning_tokens"] > 0)
            if n % 200 == 0 or n == len(todo):
                snap = meter.snapshot()
                print(
                    f"  {n}/{len(todo)} ok={counts['ok']} unparsed={counts['unparsed']} failed={counts['failed']} "
                    f"${snap['usd']:.3f} reasoning_tokens={snap['reasoning_tokens']} {time.time() - started:.0f}s",
                    flush=True,
                )
    snap = meter.snapshot()
    n_ok = counts["ok"]
    summary = {
        "rows_judged": n_ok,
        "rows_failed": counts["failed"],
        "rows_unparsed": counts["unparsed"],
        "x87_parse_rate_first_reply": (counts["x87_direct"] / n_ok) if n_ok else None,
        "x87_parse_rate_after_retry": ((n_ok - counts["unparsed"]) / n_ok) if n_ok else None,
        "benchmark_bracket_rate": (counts["benchmark_bracket"] / n_ok) if n_ok else None,
        "rows_with_reasoning_tokens": counts["reasoning_rows"],
        "reasoning_effort": reasoning_effort,
        "temperature": spec["temperature"],
        "seed": API_SEED,
        "max_tokens": API_MAX_TOKENS,
        "mean_completion_tokens": (snap["completion_tokens"] / snap["calls"]) if snap["calls"] else None,
        "seconds": time.time() - started,
        "cap_usd": meter.cap,
        "cap_hit": meter.aborted,
    } | snap
    update_manifest(f"api_{args.pass_name}", {args.arm: summary})
    if args.pass_name == "pilot":
        projected = (snap["usd"] / n_ok) * API_CALLS_PER_ARM if n_ok else None
        passed = bool(
            n_ok == len(todo)
            and counts["failed"] == 0
            and summary["x87_parse_rate_after_retry"] >= PILOT_MIN_PARSE_RATE
            and summary["benchmark_bracket_rate"] >= PILOT_MIN_PARSE_RATE
            # `none` must bill no reasoning tokens; an arm whose slug rejects `none`
            # drops to `minimal` and records the count it still bills.
            and (counts["reasoning_rows"] == 0 or reasoning_effort == "minimal")
            and projected is not None
            and projected < spec["cap_usd"]
        )
        update_manifest(
            "api_arms",
            {
                args.arm: {
                    "slug": spec["slug"],
                    "reasoning_effort": reasoning_effort,
                    "temperature": spec["temperature"],
                    "seed": API_SEED,
                    "max_tokens": API_MAX_TOKENS,
                    "cap_usd": spec["cap_usd"],
                    "pilot_rows": n_ok,
                    "pilot_reasoning_tokens": snap["reasoning_tokens"],
                    "pilot_rows_with_reasoning_tokens": counts["reasoning_rows"],
                    "pilot_x87_parse_rate": summary["x87_parse_rate_after_retry"],
                    "pilot_benchmark_bracket_rate": summary["benchmark_bracket_rate"],
                    "pilot_usd": snap["usd"],
                    "projected_usd": projected,
                    "pilot_passed": passed,
                }
            },
        )
        summary["pilot_passed"] = passed
        summary["projected_usd"] = projected
    print(json.dumps({args.arm: summary}, indent=1))
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
