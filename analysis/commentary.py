"""LLM-generated plain-English market commentary.

Feeds the day's computed analytics (sentiment score, ML regime label, and
top/bottom sector momentum) into an LLM and returns a short, descriptive market
summary. Supports Claude (Anthropic, default) and GPT (OpenAI).

The system prompt is deliberately strict: describe the current state only, no
price predictions, no buy/sell recommendations — matching this project's
"describe, don't advise" stance. SDKs are imported lazily so the rest of the app
runs without them installed; only the selected provider's package is required.
"""

from __future__ import annotations

import math

SYSTEM_PROMPT = """You are a market analyst writing a brief, plain-English end-of-day market summary for a general audience.

Structure (about 130-180 words total, two short paragraphs):
1. The overall environment — the market's risk appetite, the detected regime, and what the trend, volatility (VIX), and breadth imply, in accessible language.
2. Sector rotation — which sectors show positive vs negative momentum, and what that leaning suggests about market posture.

Hard rules:
- Describe the CURRENT state only. Do NOT predict prices or future direction.
- Do NOT give buy/sell/hold recommendations, price targets, or name specific trades.
- Ground every statement in the data provided. Do not invent numbers or facts.
- Explain jargon briefly in passing (e.g., "breadth — how many sectors are participating").
- No hype, no filler. Plain, neutral, professional tone.
- End with one short sentence noting this is descriptive analysis, not investment advice.

Output ONLY the summary text — no preamble, no headings, no bullet lists, no reasoning."""


def _fmt_pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def _fmt_num(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def build_user_prompt(ctx: dict) -> str:
    """Render the analytics context into a factual, labeled data block."""
    lines = [f"Market data as of {ctx.get('as_of', 'the latest close')}:", ""]
    lines.append(
        f"- Risk-appetite score (0-100, higher = more risk-on): "
        f"{_fmt_num(ctx.get('sentiment_score'), 1)} ({ctx.get('sentiment_label', 'n/a')})"
    )
    lines.append(f"- S&P 500 trend: {ctx.get('spy_trend', 'n/a')}")
    lines.append(f"- VIX (volatility index): {_fmt_num(ctx.get('vix'), 1)}")
    lines.append(f"- Sectors above their 50-day average (breadth): {_fmt_pct(ctx.get('breadth_50'), 0)}")
    lines.append(
        f"- Detected market regime (unsupervised ML): {ctx.get('regime', 'n/a')} "
        f"({ctx.get('regime_days', 'n/a')} trading days in this regime)"
    )
    lines.append("")
    lines.append("Strongest sectors by blended annualized momentum:")
    for name, mom, cls in ctx.get("top_sectors", []):
        lines.append(f"  - {name}: {_fmt_pct(mom, 0)} ({cls})")
    lines.append("Weakest sectors:")
    for name, mom, cls in ctx.get("bottom_sectors", []):
        lines.append(f"  - {name}: {_fmt_pct(mom, 0)} ({cls})")
    return "\n".join(lines)


def _generate_claude(user_prompt: str, model: str, api_key: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    # Opus 4.8 runs without thinking when the param is omitted; the strict
    # "output only the summary" system prompt keeps reasoning out of the text.
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if resp.stop_reason == "refusal":
        return "The model declined to generate commentary for this input."
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _generate_openai(user_prompt: str, model: str, api_key: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def generate_commentary(context: dict, provider: str = "claude",
                        model: str | None = None, api_key: str | None = None) -> str:
    """Generate a market summary from the analytics ``context``.

    provider: "claude" (default) or "openai". ``model`` defaults per provider.
    ``api_key`` may be None to let the SDK read it from the environment.
    """
    user_prompt = build_user_prompt(context)
    if provider == "claude":
        return _generate_claude(user_prompt, model or "claude-opus-4-8", api_key)
    if provider == "openai":
        return _generate_openai(user_prompt, model or "gpt-4o-mini", api_key)
    raise ValueError(f"Unknown provider: {provider!r} (expected 'claude' or 'openai')")
