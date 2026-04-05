from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import SecretStr


@dataclass(slots=True)
class RequestResult:
    ok: bool
    latency_s: float
    response_len: int
    first_token_latency_s: float = 0.0
    token_count: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def _build_llm(provider: str, temperature: float, ctx: int):
    provider = provider.strip().lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model, base_url, _ = _resolve_provider_settings(provider)

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=ctx,
        )

    if provider in {"lmstudio", "ollama-cloud"}:
        from langchain_openai import ChatOpenAI

        model, base_url, api_key = _resolve_provider_settings(provider)

        if provider == "ollama-cloud" and not api_key:
            raise RuntimeError("OLLAMA_API_KEY is required when provider=ollama-cloud")

        return ChatOpenAI(
            model_name=model,
            openai_api_base=base_url,
            openai_api_key=SecretStr(api_key),
            temperature=temperature,
            max_tokens=ctx,
            extra_body={"thinking": False},
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _resolve_provider_settings(provider: str) -> tuple[str, str, str]:
    if provider == "lmstudio":
        return (
            os.getenv("LMSTUDIO_MODEL", "qwen3.5-4b@q4_k_m"),
            os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        )

    if provider == "ollama-cloud":
        return (
            os.getenv("OLLAMA_CLOUD_MODEL", "qwen3.5:cloud"),
            os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com/v1"),
            os.getenv("OLLAMA_API_KEY", ""),
        )

    if provider == "ollama":
        return (
            os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "",
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _make_request_result(
    *,
    ok: bool,
    start_time: float,
    response_len: int,
    first_token_latency_s: float = 0.0,
    token_count: int = 0,
    tokens_per_sec: float = 0.0,
    error: str = "",
) -> RequestResult:
    return RequestResult(
        ok=ok,
        latency_s=time.perf_counter() - start_time,
        response_len=response_len,
        first_token_latency_s=first_token_latency_s,
        token_count=token_count,
        tokens_per_sec=tokens_per_sec,
        error=error,
    )


async def _one_request(
    llm: Any,
    http_client: httpx.AsyncClient,
    prompt: str,
    timeout_s: float,
    *,
    stream: bool,
    provider: str,
    temperature: float,
    ctx: int,
) -> RequestResult:
    if not stream:
        return await _invoke_langchain_request(llm, prompt=prompt, timeout_s=timeout_s)

    if provider == "ollama":
        return await _stream_ollama_request(
            http_client=http_client,
            prompt=prompt,
            timeout_s=timeout_s,
            temperature=temperature,
            ctx=ctx,
        )

    if provider in {"lmstudio", "ollama-cloud"}:
        return await _stream_openai_compatible_request(
            http_client=http_client,
            provider=provider,
            prompt=prompt,
            timeout_s=timeout_s,
            temperature=temperature,
            ctx=ctx,
        )

    raise ValueError(f"Unsupported provider: {provider}")


async def _invoke_langchain_request(
    llm: Any,
    *,
    prompt: str,
    timeout_s: float,
) -> RequestResult:
    start_time = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=timeout_s,
        )
        content = str(getattr(response, "content", response))
        return _make_request_result(
            ok=True,
            start_time=start_time,
            response_len=len(content),
        )
    except Exception as exc:  # noqa: BLE001
        return _make_request_result(
            ok=False,
            start_time=start_time,
            response_len=0,
            error=str(exc),
        )


async def _stream_ollama_request(
    http_client: httpx.AsyncClient,
    *,
    prompt: str,
    timeout_s: float,
    temperature: float,
    ctx: int,
) -> RequestResult:
    start_time = time.perf_counter()
    model, base_url, _ = _resolve_provider_settings("ollama")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": ctx},
    }

    first_token_latency = 0.0
    response_len = 0
    token_count = 0
    tokens_per_sec = 0.0

    try:
        async with http_client.stream(
            "POST",
            f"{base_url}/api/generate",
            json=payload,
            timeout=httpx.Timeout(timeout_s),
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                piece = str(chunk.get("response", ""))
                if piece and first_token_latency == 0.0:
                    first_token_latency = time.perf_counter() - start_time
                response_len += len(piece)

                if not chunk.get("done"):
                    continue

                eval_count = int(chunk.get("eval_count") or 0)
                eval_duration_ns = int(chunk.get("eval_duration") or 0)
                token_count = eval_count
                if eval_count > 0 and eval_duration_ns > 0:
                    tokens_per_sec = eval_count / (eval_duration_ns / 1_000_000_000)

        return _make_request_result(
            ok=True,
            start_time=start_time,
            response_len=response_len,
            first_token_latency_s=first_token_latency,
            token_count=token_count,
            tokens_per_sec=tokens_per_sec,
        )
    except Exception as exc:  # noqa: BLE001
        return _make_request_result(
            ok=False,
            start_time=start_time,
            response_len=0,
            error=str(exc),
        )


async def _stream_openai_compatible_request(
    http_client: httpx.AsyncClient,
    *,
    provider: str,
    prompt: str,
    timeout_s: float,
    temperature: float,
    ctx: int,
) -> RequestResult:
    start_time = time.perf_counter()
    model, base_url, api_key = _resolve_provider_settings(provider)

    if provider == "ollama-cloud" and not api_key:
        raise RuntimeError("OLLAMA_API_KEY is required when provider=ollama-cloud")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": temperature,
        "max_tokens": ctx,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    first_token_latency = 0.0
    response_len = 0
    token_count = 0
    tokens_per_sec = 0.0

    try:
        async with http_client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
        ) as resp:
            resp.raise_for_status()

            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    continue

                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices and first_token_latency == 0.0:
                    first_token_latency = time.perf_counter() - start_time

                # Try to accumulate content for response length
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = str(delta.get("content", ""))
                    response_len += len(piece)

                usage = chunk.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens") or 0)
                if completion_tokens > 0:
                    token_count = completion_tokens
                    elapsed = time.perf_counter() - start_time
                    if elapsed > 0:
                        tokens_per_sec = completion_tokens / elapsed

        return _make_request_result(
            ok=True,
            start_time=start_time,
            response_len=response_len,
            first_token_latency_s=first_token_latency,
            token_count=token_count,
            tokens_per_sec=tokens_per_sec,
        )
    except Exception as exc:  # noqa: BLE001
        return _make_request_result(
            ok=False,
            start_time=start_time,
            response_len=0,
            error=str(exc),
        )


def _parse_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for part in raw.split(","):
        value = int(part.strip())
        if value <= 0:
            raise ValueError("Concurrency level must be positive")
        levels.append(value)
    return levels


def _print_kv_block(title: str, rows: list[tuple[str, str]]) -> None:
    print(title)
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")


async def _run_level(
    *,
    llm: Any,
    http_client: httpx.AsyncClient,
    prompt: str,
    concurrency: int,
    requests_per_level: int,
    timeout_s: float,
    stream: bool,
    provider: str,
    temperature: float,
    ctx: int,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _single_request() -> RequestResult:
        async with semaphore:
            return await _one_request(
                llm=llm,
                http_client=http_client,
                prompt=prompt,
                timeout_s=timeout_s,
                stream=stream,
                provider=provider,
                temperature=temperature,
                ctx=ctx,
            )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(_single_request() for _ in range(requests_per_level))
    )
    elapsed = time.perf_counter() - started
    completed = sum(1 for result in results if result.ok)
    total_tokens = sum(result.token_count for result in results if result.ok)
    throughput = completed / elapsed if elapsed > 0 else 0.0
    print(
        f"level={concurrency} requests={requests_per_level} "
        f"ok={completed}/{requests_per_level} wall={elapsed:.2f}s "
        f"throughput={throughput:.2f} req/s tokens={total_tokens}"
    )
    return list(results)


def _print_summary(level: int, results: list[RequestResult], *, label: str) -> None:
    ok_results = [result for result in results if result.ok]
    failed_results = [result for result in results if not result.ok]

    latencies = [result.latency_s for result in ok_results]
    first_tokens = [
        result.first_token_latency_s
        for result in ok_results
        if result.first_token_latency_s > 0
    ]
    token_rates = [
        result.tokens_per_sec for result in ok_results if result.tokens_per_sec > 0
    ]
    response_lengths = [result.response_len for result in ok_results]

    total = len(results)
    ok_count = len(ok_results)
    fail_count = len(failed_results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_first_token = sum(first_tokens) / len(first_tokens) if first_tokens else 0.0
    avg_token_rate = sum(token_rates) / len(token_rates) if token_rates else 0.0

    print(f"\n[{label}] concurrency={level}")
    _print_kv_block(
        "Summary",
        [
            ("requests", str(total)),
            ("ok", str(ok_count)),
            ("failed", str(fail_count)),
            ("avg latency", f"{avg_latency:.2f}s"),
            ("p50 latency", f"{_percentile(latencies, 0.50):.2f}s"),
            ("p95 latency", f"{_percentile(latencies, 0.95):.2f}s"),
            ("p99 latency", f"{_percentile(latencies, 0.99):.2f}s"),
            ("avg first token", f"{avg_first_token:.2f}s"),
            ("p95 first token", f"{_percentile(first_tokens, 0.95):.2f}s"),
            ("avg token TPS", f"{avg_token_rate:.2f}"),
            (
                "avg response len",
                f"{(sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0:.0f}",
            ),
        ],
    )

    if failed_results:
        print("Failures")
        for index, result in enumerate(failed_results[:3], start=1):
            print(f"  {index}. {result.error}")


async def _amain(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)

    provider = args.provider or os.getenv("LLM_PROVIDER", "lmstudio")

    levels = _parse_levels(args.levels)
    _print_kv_block(
        "LLM stress test",
        [
            ("provider", provider),
            ("levels", str(levels)),
            ("requests/level", str(args.requests_per_level)),
            ("timeout/request", f"{args.timeout_s:.1f}s"),
            ("ctx", str(args.ctx)),
            ("stream", str(args.stream)),
            ("warmup", str(args.warmup)),
        ],
    )

    llm = _build_llm(provider=provider, temperature=args.temperature, ctx=args.ctx)

    async with httpx.AsyncClient() as http_client:
        # Cold-start samples are executed first and reported separately.
        if args.warmup > 0:
            print("\n--- Cold-start phase ---")
            cold_results = await _run_level(
                llm=llm,
                http_client=http_client,
                prompt=args.prompt,
                concurrency=1,
                requests_per_level=args.warmup,
                timeout_s=args.timeout_s,
                stream=args.stream,
                provider=provider,
                temperature=args.temperature,
                ctx=args.ctx,
            )
            _print_summary(1, cold_results, label="Cold start")

        print("\n--- Steady-state phase ---")
        for level in levels:
            results = await _run_level(
                llm=llm,
                http_client=http_client,
                prompt=args.prompt,
                concurrency=level,
                requests_per_level=args.requests_per_level,
                timeout_s=args.timeout_s,
                stream=args.stream,
                provider=provider,
                temperature=args.temperature,
                ctx=args.ctx,
            )
            _print_summary(level, results, label="Steady state")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test local/cloud LLM endpoints used by this project. "
            "Useful for picking safe concurrency for lyrics-composer/validator."
        )
    )
    parser.add_argument(
        "--provider",
        default="",
        help="llm provider: ollama | lmstudio | ollama-cloud (defaults to LLM_PROVIDER/.env)",
    )
    parser.add_argument(
        "--levels",
        default="1,2,3",
        help="comma-separated concurrency levels, e.g. 1,2,4",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=6,
        help="number of requests to send at each concurrency level",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="timeout for each request in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        help="sampling temperature",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=int(os.getenv("LLM_CTX", "8192")),
        help="context size passed to provider",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "请写一段约 100 字粤语歌词草稿，语气积极，注意押韵。"
            "只输出歌词正文，不要解释。"
        ),
        help="prompt used for each test request",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="use streaming mode (ollama) and report first-token latency + token TPS",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup requests (excluded from metrics)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
