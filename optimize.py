from __future__ import annotations

import html
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from ranker import (
    JST, ROOT, fetch_chart, flip_rate, load_tickers, pct_change, percentiles,
    price_efficiency, realized_volatility, same_time_relative_volume, split_days,
)


COST_BPS = 10.0
TOP_K = 3
ANNEALING_TRIALS = 3000
SEED = 69069
FEATURES = [
    "relative_strength", "momentum_15m", "relative_volume", "range_position",
    "efficiency", "low_flip", "turnover", "volatility_sweet",
    "negative_day", "recovery", "near_vwap",
]
OUTPUT_JSON = ROOT / "docs" / "simulation.json"
OUTPUT_HTML = ROOT / "docs" / "simulation.html"


@dataclass
class Sample:
    date: str
    ts: int
    code: str
    name: str
    features: list[float]
    future_net_pct: float


def sessions(rows: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    morning = [r for r in rows if r["minute"] < 12 * 60]
    afternoon = [r for r in rows if r["minute"] >= 12 * 60]
    return [s for s in (morning, afternoon) if s]


def benchmark_momentum_map(chart: dict) -> dict[int, float]:
    result = {}
    for rows in split_days(chart).values():
        for session in sessions(rows):
            for i in range(12, len(session) - 12, 12):
                result[int(session[i]["ts"])] = pct_change(
                    session[i]["close"], session[i - 12]["close"]
                )
    return result


def raw_samples(item: dict[str, str], chart: dict, market: dict[int, float]) -> list[dict]:
    days = split_days(chart)
    ordered = sorted(days)
    result = []
    for day_pos, day_key in enumerate(ordered):
        if day_pos < 3:
            continue
        day_rows = days[day_key]
        previous = [days[d] for d in ordered[:day_pos]]
        prev_close = previous[-1][-1]["close"]
        for session in sessions(day_rows):
            for i in range(12, len(session) - 12, 12):
                entry = session[i]
                ts = int(entry["ts"])
                if ts not in market:
                    continue
                recent = session[i - 12:i + 1]
                prefix = [r for r in day_rows if r["ts"] <= entry["ts"]]
                closes = [r["close"] for r in recent]
                last_15 = closes[-4:]
                close = entry["close"]
                high = max(r["high"] for r in prefix)
                low = min(r["low"] for r in prefix)
                span = max(high - low, close * 1e-9)
                volume = sum(r["volume"] for r in prefix)
                typical_value = sum(
                    ((r["high"] + r["low"] + r["close"]) / 3) * r["volume"]
                    for r in prefix
                )
                vwap = typical_value / volume if volume else close
                day_change = pct_change(close, prev_close)
                momentum = pct_change(closes[-1], closes[0])
                result.append({
                    "date": day_key, "ts": ts, "code": item["code"],
                    "name": item["name"],
                    "raw": {
                        "relative_strength": momentum - market[ts],
                        "momentum_15m": pct_change(last_15[-1], last_15[0]),
                        "relative_volume": same_time_relative_volume(prefix, previous),
                        "range_position": (close - low) / span,
                        "efficiency": price_efficiency(closes),
                        "low_flip": 1 - flip_rate(closes),
                        "turnover": math.log1p(close * volume / 1_000_000),
                        "realized_vol": realized_volatility(closes),
                        "negative_day": -day_change,
                        "recovery": pct_change(close, low),
                        "near_vwap": -abs(pct_change(close, vwap)),
                    },
                    "future_net_pct": pct_change(session[i + 12]["close"], close)
                    - COST_BPS / 100,
                })
    return result


def rank_cross_section(rows: list[dict]) -> list[Sample]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["ts"]].append(row)
    samples = []
    rank_fields = [f for f in FEATURES if f != "volatility_sweet"] + ["realized_vol"]
    for group in grouped.values():
        ranks = {
            field: percentiles([row["raw"][field] for row in group])
            for field in rank_fields
        }
        for i, row in enumerate(group):
            values = []
            for feature in FEATURES:
                if feature == "volatility_sweet":
                    p = ranks["realized_vol"][i]
                    values.append(max(0.0, 1 - abs(p - 0.65) / 0.65))
                else:
                    values.append(ranks[feature][i])
            samples.append(Sample(
                row["date"], row["ts"], row["code"], row["name"],
                values, row["future_net_pct"],
            ))
    return samples


def evaluate(samples: list[Sample], weights: list[float], threshold: float) -> dict:
    grouped: dict[int, list[tuple[float, Sample]]] = defaultdict(list)
    for sample in samples:
        score = sum(w * x for w, x in zip(weights, sample.features))
        if score >= threshold:
            grouped[sample.ts].append((score, sample))
    event_returns, trades = [], []
    for ts in sorted(grouped):
        chosen = sorted(grouped[ts], key=lambda x: x[0], reverse=True)[:TOP_K]
        if not chosen:
            continue
        event_returns.append(mean(sample.future_net_pct for _, sample in chosen))
        trades.extend(sample.future_net_pct for _, sample in chosen)
    equity, peak, max_drawdown = 1.0, 1.0, 0.0
    for ret in event_returns:
        equity *= 1 + ret / 100
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    avg = mean(trades) if trades else 0.0
    sd = pstdev(event_returns) if len(event_returns) > 1 else 0.0
    sharpe = mean(event_returns) / sd * math.sqrt(3 * 252) if sd else 0.0
    return {
        "trades": len(trades), "events": len(event_returns),
        "avg_net_pct": avg,
        "win_rate_pct": sum(r > 0 for r in trades) / len(trades) * 100 if trades else 0.0,
        "total_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": max_drawdown, "annualized_sharpe": sharpe,
    }


def objective(result: dict) -> float:
    if result["trades"] < 80:
        return -100 + result["trades"] / 100
    return result["annualized_sharpe"] - 0.12 * result["max_drawdown_pct"]


def random_weights(rng: random.Random) -> list[float]:
    values = [-math.log(max(rng.random(), 1e-12)) for _ in FEATURES]
    total = sum(values)
    return [v / total for v in values]


def anneal(train: list[Sample]) -> tuple[list[float], float, dict]:
    rng = random.Random(SEED)
    weights = random_weights(rng)
    threshold = 0.62
    current = evaluate(train, weights, threshold)
    current_obj = objective(current)
    best = (weights[:], threshold, current, current_obj)
    for step in range(ANNEALING_TRIALS):
        temperature = 0.25 * (0.003 / 0.25) ** (step / ANNEALING_TRIALS)
        candidate = weights[:]
        a, b = rng.sample(range(len(candidate)), 2)
        transfer = min(candidate[a], abs(rng.gauss(0, 0.08)))
        candidate[a] -= transfer
        candidate[b] += transfer
        candidate_threshold = min(0.85, max(0.50, threshold + rng.gauss(0, 0.025)))
        result = evaluate(train, candidate, candidate_threshold)
        obj = objective(result)
        if obj > current_obj or rng.random() < math.exp((obj - current_obj) / temperature):
            weights, threshold, current, current_obj = candidate, candidate_threshold, result, obj
        if obj > best[3]:
            best = (candidate[:], candidate_threshold, result, obj)
    return best[0], best[1], best[2]


def fmt(result: dict) -> dict:
    return {
        "trades": result["trades"], "events": result["events"],
        "avg_net_pct": round(result["avg_net_pct"], 4),
        "win_rate_pct": round(result["win_rate_pct"], 1),
        "total_return_pct": round(result["total_return_pct"], 2),
        "max_drawdown_pct": round(result["max_drawdown_pct"], 2),
        "annualized_sharpe": round(result["annualized_sharpe"], 2),
    }


def render(report: dict) -> str:
    rows = "".join(
        f"<tr><th>{label}</th><td>{data['trades']}</td><td>{data['avg_net_pct']:+.4f}%</td>"
        f"<td>{data['win_rate_pct']:.1f}%</td><td>{data['total_return_pct']:+.2f}%</td>"
        f"<td>{data['max_drawdown_pct']:.2f}%</td><td>{data['annualized_sharpe']:.2f}</td></tr>"
        for label, data in (("学習", report["train"]), ("検証", report["validation"]), ("未使用テスト", report["test"]))
    )
    weights = "".join(
        f"<li><span>{html.escape(name)}</span><b>{weight*100:.1f}%</b></li>"
        for name, weight in sorted(report["weights"].items(), key=lambda x: x[1], reverse=True)
    )
    verdict_class = "ok" if report["paper_ready"] else "wait"
    verdict = "仮想運用へ進む条件を通過" if report["paper_ready"] else "条件未達：最適化結果を実売買に使わない"
    return f"""<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>焼きなまし仮想運用レポート</title><style>
body{{margin:0;background:#07111f;color:#e8f0f7;font-family:system-ui,sans-serif}}main{{max-width:1000px;margin:auto;padding:20px}}a{{color:#76b7ff}}.box{{background:#101d2d;border:1px solid #20354b;border-radius:14px;padding:16px;margin:15px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:700px}}th,td{{padding:10px;border-bottom:1px solid #20354b;text-align:right}}th:first-child{{text-align:left}}.verdict{{font-size:20px;font-weight:800;padding:14px;border-radius:10px}}.ok{{background:#153d31;color:#75e7ba}}.wait{{background:#3b2c20;color:#ffc179}}ul{{padding:0;list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}}li{{display:flex;justify-content:space-between;background:#16263a;padding:10px;border-radius:8px}}p{{color:#a6b6c6;line-height:1.65}}</style></head><body><main><p><a href='index.html'>←ランキングへ</a></p><h1>焼きなまし仮想運用レポート</h1><p>更新：{html.escape(report['generated_at'])}／過去{report['trading_days']}営業日／試行{report['trials']:,}回／往復コスト仮定{report['cost_bps']:.1f}bp</p><div class='verdict {verdict_class}'>{verdict}</div><div class='box'><table><tr><th>期間</th><th>取引数</th><th>1取引平均</th><th>勝率</th><th>累積</th><th>最大DD</th><th>年率Sharpe</th></tr>{rows}</table></div><div class='box'><h2>選ばれた評価ウェイト</h2><ul>{weights}</ul><p>閾値：{report['threshold']:.3f}／各時点の上位{TOP_K}銘柄／1時間保有。焼きなましは学習期間だけを見ています。学習80取引以上、検証と未使用テストは各20取引以上で、3期間すべてがコスト控除後プラスの場合だけ仮想運用条件通過とします。データ取得エラーが1件でもあれば不合格です。</p></div><div class='box'><h2>重要な制限</h2><p>これは過去データ上のシミュレーションで、明日の利益を保証しません。現在の主要30銘柄だけを遡るため生存者バイアスがあり、無料データには遅延・欠損があります。最適化を何度も試すほど偶然の好成績を拾うため、実資金ではなく前向きな仮想運用で再検証してください。</p></div></main></body></html>"""


def main() -> None:
    market = benchmark_momentum_map(fetch_chart("1306", "60d"))
    raw, errors = [], []
    for item in load_tickers():
        try:
            raw.extend(raw_samples(item, fetch_chart(item["code"], "60d"), market))
        except Exception as exc:
            errors.append(f"{item['code']}: {exc}")
    samples = rank_cross_section(raw)
    dates = sorted({s.date for s in samples})
    first = max(1, int(len(dates) * 0.60))
    second = max(first + 1, int(len(dates) * 0.80))
    train_dates, validation_dates, test_dates = set(dates[:first]), set(dates[first:second]), set(dates[second:])
    train = [s for s in samples if s.date in train_dates]
    validation = [s for s in samples if s.date in validation_dates]
    test = [s for s in samples if s.date in test_dates]
    weights, threshold, train_result = anneal(train)
    validation_result = evaluate(validation, weights, threshold)
    test_result = evaluate(test, weights, threshold)
    paper_ready = (
        not errors
        and train_result["trades"] >= 80
        and validation_result["trades"] >= 20
        and test_result["trades"] >= 20
        and train_result["avg_net_pct"] > 0
        and validation_result["avg_net_pct"] > 0
        and test_result["avg_net_pct"] > 0
        and train_result["total_return_pct"] > 0
        and validation_result["total_return_pct"] > 0
        and test_result["total_return_pct"] > 0
        and train_result["annualized_sharpe"] > 0
        and validation_result["annualized_sharpe"] > 0
        and test_result["annualized_sharpe"] > 0
    )
    report = {
        "generated_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M JST"),
        "trading_days": len(dates), "samples": len(samples), "trials": ANNEALING_TRIALS,
        "cost_bps": COST_BPS, "top_k": TOP_K, "threshold": round(threshold, 4),
        "weights": {name: round(value, 5) for name, value in zip(FEATURES, weights)},
        "train": fmt(train_result), "validation": fmt(validation_result),
        "test": fmt(test_result), "paper_ready": paper_ready, "errors": errors,
        "split": {"train": sorted(train_dates), "validation": sorted(validation_dates), "test": sorted(test_dates)},
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_HTML.write_text(render(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("trading_days", "samples", "threshold", "train", "validation", "test", "paper_ready", "errors")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
