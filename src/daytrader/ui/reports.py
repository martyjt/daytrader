"""Markdown export of research artifacts.

Turns in-memory research results (shadow tournaments, portfolio backtests,
regime snapshots) into Markdown so users can paste them into PR
descriptions, journals, or chat channels. Everything is pure formatting
— no DB writes, no network.
"""

from __future__ import annotations

from datetime import datetime


def tournament_to_markdown(report) -> str:
    """Render a ``ShadowTournamentReport`` as a Markdown block."""
    lines: list[str] = []
    lines.append(f"## Shadow Tournament — {report.target_symbol} {report.target_timeframe}")
    lines.append("")
    lines.append(
        f"- **Window**: {_fmt_dt(report.window_start)} → {_fmt_dt(report.window_end)}"
    )
    lines.append(f"- **Primary (incumbent)**: `{report.primary_algo_id}`")
    lines.append(f"- **Candidates tested**: {len(report.candidates)}")
    lines.append(f"- **Winners**: {len(report.winners)}")
    lines.append("")
    lines.append(
        "| Algorithm | OOS Sharpe | OOS Return % | Max DD % | # Trades | Fold Stability | Verdict |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|"
    )
    for c in report.ranked():
        verdict = (
            "**PRIMARY**" if c.is_primary
            else ("**WINNER**" if c.beat_primary else "—")
        )
        lines.append(
            f"| {c.algo_name} | {c.sharpe:+.2f} | {c.net_return_pct:+.2f} | "
            f"{c.max_drawdown_pct:+.2f} | {c.num_trades} | "
            f"{int(c.stability_score * 100)}% | {verdict} |"
        )
    if report.winners:
        lines.append("")
        lines.append(
            f"_Candidates passing both Sharpe + ≥50% fold stability: "
            f"{', '.join(report.winners)}_"
        )
    return "\n".join(lines)


def portfolio_to_markdown(report) -> str:
    """Render a ``PortfolioBacktestResult`` as a Markdown block."""
    lines: list[str] = []
    lines.append("## Portfolio Backtest")
    lines.append("")
    lines.append(
        f"- **Total capital**: ${report.initial_total_capital:,.0f} "
        f"→ ${report.final_total_equity:,.0f}"
    )
    lines.append(
        f"- **Portfolio return**: {report.portfolio_return_pct:+.2f}% · "
        f"Sharpe {report.portfolio_sharpe:.2f} · "
        f"Max DD {report.portfolio_max_drawdown_pct:.2f}%"
    )
    if report.best_symbol and report.worst_symbol:
        lines.append(
            f"- **Best / Worst**: {report.best_symbol.symbol} "
            f"(Sharpe {report.best_symbol.sharpe:+.2f}) / "
            f"{report.worst_symbol.symbol} "
            f"(Sharpe {report.worst_symbol.sharpe:+.2f})"
        )
    lines.append("")
    lines.append(
        "| Symbol | Sharpe | Return % | Max DD % | # Trades | Final $ | Status |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---|"
    )
    for s in report.symbols:
        status = f"error: {s.error}" if s.error else "ok"
        lines.append(
            f"| {s.symbol} | {s.sharpe:+.2f} | {s.net_return_pct:+.2f} | "
            f"{s.max_drawdown_pct:+.2f} | {s.num_trades} | "
            f"${s.final_equity:,.0f} | {status} |"
        )
    return "\n".join(lines)


def regime_to_markdown(snapshot) -> str:
    """Render a ``RegimeSnapshot`` as a brief Markdown status line."""
    if snapshot.status != "ok":
        return f"**Regime**: _{snapshot.status}_ ({snapshot.message or '—'})"
    probs = ", ".join(
        f"{k}: {int(round(v * 100))}%"
        for k, v in snapshot.probabilities.items()
    )
    return (
        f"**Regime** (pulse: {snapshot.pulse_symbol} {snapshot.pulse_timeframe}): "
        f"**{snapshot.regime.upper()}** — {probs}"
    )


def _fmt_dt(dt) -> str:
    if not isinstance(dt, datetime):
        return str(dt)
    return dt.strftime("%Y-%m-%d")
