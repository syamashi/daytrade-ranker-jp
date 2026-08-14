from __future__ import annotations

import csv
import html
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
TICKERS_FILE = ROOT / "tickers.csv"
OUTPUT_FILE = ROOT / "docs" / "index.html"
JST = ZoneInfo("Asia/Tokyo")


def load_tickers() -> list[dict[str, str]]:
    with TICKERS_FILE.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fetch_chart(code: str) -> dict:
    symbol = f"{code}.T"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    query = urllib.parse.urlencode(
        {"interval": "5m", "range": "5d", "includePrePost": "false"}
    )
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0 daytrade-ranker/0.2"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)["chart"]["result"]
    if not result:
        raise ValueError(f"No data: {symbol}")
    return result[0]


def split_days(chart: dict) -> dict[str, list[dict[str, float]]]:
    quote = chart["indicators"]["quote"][0]
    days: dict[str, list[dict[str, float]]] = {}
    for ts, close, high, low, volume in zip(
        chart["timestamp"], quote["close"], quote["high"], quote["low"], quote["volume"]
    ):
        if None in (close, high, low, volume):
            continue
        dt = datetime.fromtimestamp(ts, JST)
        day = dt.date().isoformat()
        days.setdefault(day, []).append(
            {
                "ts": float(ts), "minute": float(dt.hour * 60 + dt.minute),
                "close": float(close), "high": float(high), "low": float(low),
                "volume": float(volume),
            }
        )
    return days


def contiguous_tail(rows: list[dict[str, float]], limit: int = 13) -> list[dict[str, float]]:
    """Return at most one continuous trading hour; do not bridge the lunch break."""
    tail = [rows[-1]]
    for row in reversed(rows[:-1]):
        if tail[0]["ts"] - row["ts"] > 10 * 60:
            break
        tail.insert(0, row)
        if len(tail) >= limit:
            break
    return tail


def pct_change(new: float, old: float) -> float:
    return (new / old - 1) * 100 if old else 0.0


def price_efficiency(closes: list[float]) -> float:
    """Kaufman-style efficiency: net move / total path, 0=choppy, 1=straight."""
    if len(closes) < 2:
        return 0.0
    path = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
    return abs(closes[-1] - closes[0]) / path if path else 0.0


def flip_rate(closes: list[float]) -> float:
    """Share of non-zero 5-minute returns whose direction flips."""
    signs = []
    for a, b in zip(closes, closes[1:]):
        diff = b - a
        if diff:
            signs.append(1 if diff > 0 else -1)
    if len(signs) < 2:
        return 0.0
    return sum(a != b for a, b in zip(signs, signs[1:])) / (len(signs) - 1)


def realized_volatility(closes: list[float]) -> float:
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    return math.sqrt(sum(r * r for r in returns)) * 100


def same_time_relative_volume(
    today: list[dict[str, float]], previous_days: list[list[dict[str, float]]]
) -> float:
    cutoff = today[-1]["minute"]
    current = sum(r["volume"] for r in today)
    baselines = [
        sum(r["volume"] for r in day if r["minute"] <= cutoff)
        for day in previous_days
    ]
    baselines = [v for v in baselines if v > 0]
    baseline = mean(baselines) if baselines else current
    return current / baseline if baseline else 1.0


def benchmark_return(chart: dict) -> tuple[float, int]:
    days = split_days(chart)
    today = days[sorted(days)[-1]]
    recent = contiguous_tail(today)
    minutes = round((recent[-1]["ts"] - recent[0]["ts"]) / 60)
    return pct_change(recent[-1]["close"], recent[0]["close"]), minutes


def metrics(item: dict[str, str], chart: dict, market_momentum: float) -> dict:
    days = split_days(chart)
    ordered = sorted(days)
    if len(ordered) < 2:
        raise ValueError("Not enough trading days")
    today = days[ordered[-1]]
    previous_days = [days[d] for d in ordered[:-1]]
    recent = contiguous_tail(today)
    if len(today) < 3 or len(recent) < 2:
        raise ValueError("Not enough intraday bars")

    closes = [r["close"] for r in recent]
    last_15 = closes[-4:]
    close = today[-1]["close"]
    prev_close = previous_days[-1][-1]["close"]
    day_high = max(r["high"] for r in today)
    day_low = min(r["low"] for r in today)
    day_range = max(day_high - day_low, close * 1e-9)
    volume_today = sum(r["volume"] for r in today)
    typical_value = sum(
        ((r["high"] + r["low"] + r["close"]) / 3) * r["volume"] for r in today
    )
    vwap = typical_value / volume_today if volume_today else close
    momentum = pct_change(closes[-1], closes[0])
    momentum_15m = pct_change(last_15[-1], last_15[0])

    return {
        "code": item["code"], "name": item["name"], "price": close,
        "asof": datetime.fromtimestamp(today[-1]["ts"], JST).strftime("%m/%d %H:%M"),
        "bars": len(recent),
        "window_minutes": round((recent[-1]["ts"] - recent[0]["ts"]) / 60),
        "day_change": pct_change(close, prev_close),
        "momentum_1h": momentum, "momentum_15m": momentum_15m,
        "relative_strength": momentum - market_momentum,
        "vwap_gap": pct_change(close, vwap),
        "relative_volume": same_time_relative_volume(today, previous_days),
        "turnover_million": close * volume_today / 1_000_000,
        "range_pct": day_range / prev_close * 100,
        "range_position": (close - day_low) / day_range,
        "recovery_from_low": pct_change(close, day_low),
        "drawdown_from_high": pct_change(close, day_high),
        "efficiency": price_efficiency(closes), "flip_rate": flip_rate(closes),
        "realized_vol": realized_volatility(closes),
        "confidence": min(1.0, len(recent) / 10) * min(1.0, len(previous_days) / 3),
    }


def percentiles(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.5] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = ((start + end) / 2) / (len(values) - 1)
        for pos in range(start, end + 1):
            result[order[pos]] = rank
        start = end + 1
    return result


def score_all(rows: list[dict]) -> list[dict]:
    fields = [
        "relative_strength", "momentum_15m", "relative_volume", "range_position",
        "efficiency", "flip_rate", "turnover_million", "realized_vol",
        "day_change", "recovery_from_low", "vwap_gap",
    ]
    ranks = {field: percentiles([r[field] for r in rows]) for field in fields}

    for i, row in enumerate(rows):
        p = {field: ranks[field][i] for field in fields}
        volatility_sweet_spot = max(0.0, 1 - abs(p["realized_vol"] - 0.65) / 0.65)
        low_chop = 1 - p["flip_rate"]

        trend = 100 * (
            0.22 * p["relative_strength"] + 0.10 * p["momentum_15m"]
            + 0.15 * p["relative_volume"] + 0.13 * p["range_position"]
            + 0.15 * p["efficiency"] + 0.10 * low_chop
            + 0.10 * p["turnover_million"] + 0.05 * volatility_sweet_spot
        )
        if row["relative_strength"] <= 0:
            trend *= 0.55
        if row["momentum_15m"] <= 0:
            trend *= 0.75
        if row["vwap_gap"] <= 0:
            trend *= 0.80
        trend -= max(0.0, row["day_change"] - 8) * 3
        trend -= max(0.0, row["vwap_gap"] - 3) * 4

        near_vwap = max(0.0, 1 - abs(row["vwap_gap"]) / max(row["range_pct"], 0.25))
        rebound = 100 * (
            0.20 * (1 - p["day_change"]) + 0.22 * p["momentum_15m"]
            + 0.15 * p["recovery_from_low"] + 0.10 * p["range_position"]
            + 0.10 * p["efficiency"] + 0.08 * low_chop
            + 0.08 * p["relative_volume"] + 0.04 * p["turnover_million"]
            + 0.03 * near_vwap
        )
        if row["day_change"] >= 0:
            rebound *= 0.45
        if row["momentum_15m"] <= 0:
            rebound *= 0.50

        tradability = 100 * (
            0.35 * p["turnover_million"] + 0.20 * p["relative_volume"]
            + 0.20 * volatility_sweet_spot + 0.15 * p["efficiency"] + 0.10 * low_chop
        )
        best = max(trend, rebound)
        overall = best * (0.65 + 0.35 * tradability / 100) * (0.75 + 0.25 * row["confidence"])

        row["trend_score"] = round(max(0, min(100, trend)), 1)
        row["rebound_score"] = round(max(0, min(100, rebound)), 1)
        row["tradability_score"] = round(max(0, min(100, tradability)), 1)
        row["score"] = round(max(0, min(100, overall)), 1)
        if best < 55:
            row["signal"] = "様子見"
        elif trend >= rebound:
            row["signal"] = "上昇継続"
        else:
            row["signal"] = "反発"
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def render(
    rows: list[dict], errors: list[str], market_momentum: float, market_window: int
) -> str:
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    table = "".join(
        f"<tr><td>{i}</td><td><b>{html.escape(r['code'])}</b><br><small>{html.escape(r['name'])}</small></td>"
        f"<td><span class='tag {('trend' if r['signal']=='上昇継続' else 'bounce' if r['signal']=='反発' else '')}'>{r['signal']}</span></td>"
        f"<td class='score'>{r['score']:.1f}</td><td>{r['trend_score']:.1f}</td><td>{r['rebound_score']:.1f}</td>"
        f"<td>{r['price']:,.1f}</td><td class='{'up' if r['day_change'] >= 0 else 'down'}'>{r['day_change']:+.2f}%</td>"
        f"<td>{r['relative_strength']:+.2f}%</td><td>{r['momentum_15m']:+.2f}%</td>"
        f"<td>{r['relative_volume']:.2f}x</td><td>{r['range_pct']:.2f}%</td>"
        f"<td>{r['range_position']*100:.0f}%</td><td>{r['efficiency']*100:.0f}</td>"
        f"<td>{r['flip_rate']*100:.0f}%</td><td>{r['vwap_gap']:+.2f}%</td></tr>"
        for i, r in enumerate(rows, 1)
    )
    factors = [
        ("相対強度", "同じ1時間の市場ETFより強いか。地合いだけの上昇を除く。"),
        ("15分・1時間モメンタム", "直近方向の継続性。逆向きなら継続点を減点。"),
        ("同時刻出来高比", "過去日と同じ時刻までの出来高を比較。単なる日中季節性を抑える。"),
        ("VWAP乖離", "平均的な約定価格より上か。ただし離れすぎは過熱として減点。"),
        ("日中レンジ位置", "安値0%・高値100%。上側維持か、安値から戻ったかを見る。"),
        ("値幅率・実現ボラ", "高値−安値を株価で割る。絶対円では比較しない。極端な荒さも低さも避ける。"),
        ("価格効率", "純移動÷全移動。100に近いほど一方向、0に近いほど往復が多い。"),
        ("方向反転率", "5分足の上下が入れ替わる頻度。継続狙いでは低い方を評価。"),
        ("売買代金", "約定しやすさの代理。薄商いによる見かけのシグナルを抑える。"),
    ]
    factor_cards = "".join(
        f"<article><b>{html.escape(name)}</b><span>{html.escape(desc)}</span></article>"
        for name, desc in factors
    )
    payload = html.escape(json.dumps(rows, ensure_ascii=False))
    return f"""<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>日本株デイトレ候補ランキング</title><style>
:root{{--bg:#07111f;--panel:#101d2d;--text:#e8f0f7;--muted:#91a4b7;--accent:#54d6a0;--blue:#65aaff;--red:#ff6f7d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:20px}}
h1{{font-size:clamp(23px,4vw,36px);margin:0 0 5px}}h2{{margin-top:28px}}p{{color:var(--muted)}}.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
.pill,.tag{{display:inline-block;padding:5px 9px;border-radius:999px;background:#1b2a3c;color:#c7d5e2}}.tag.trend{{background:#163d32;color:#75e7ba}}.tag.bounce{{background:#3b263e;color:#eda4ff}}
.card{{background:var(--panel);border:1px solid #20354b;border-radius:14px;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:1280px}}
th,td{{padding:10px 11px;border-bottom:1px solid #20354b;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#15263a;color:#b9c8d6;font-size:12px}}
th:nth-child(2),td:nth-child(2){{text-align:left}}.score{{font-size:18px;color:var(--accent);font-weight:800}}.up{{color:var(--red)}}.down{{color:var(--blue)}}small,article span{{color:var(--muted)}}
.factors{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px}}article{{padding:14px;background:var(--panel);border:1px solid #20354b;border-radius:12px}}article b,article span{{display:block}}article span{{font-size:13px;margin-top:5px;line-height:1.5}}
.notice{{margin-top:16px;padding:14px;background:#17202b;border-radius:10px;font-size:13px;line-height:1.65}}a{{color:#7ab8ff}}@media(max-width:600px){{main{{padding:12px}}}}
</style></head><body><main><h1>日本株デイトレ候補ランキング</h1><p>最終更新：{now}　データ時点：{html.escape(rows[0]['asof']) if rows else '-'}　監視：{len(rows)}銘柄</p>
<div class='summary'><span class='pill'>市場直近{market_window}分：{market_momentum:+.2f}%</span><span class='pill'>総合＝シグナル強度×売買しやすさ</span><span class='pill'>上昇継続と反発を別採点</span></div>
<div class='card'><table><thead><tr><th>#</th><th>銘柄</th><th>判定</th><th>総合</th><th>継続</th><th>反発</th><th>現在値</th><th>前日比</th><th>市場超過・直近</th><th>15分</th><th>出来高比</th><th>値幅率</th><th>レンジ位置</th><th>価格効率</th><th>反転頻度</th><th>VWAP乖離</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>評価に使う要素</h2><div class='factors'>{factor_cards}</div>
<div class='notice'><b>読み方：</b>「上がったものが上がり続ける」候補は、相対強度・出来高・高値圏維持・低い反転頻度を重視します。「今日下がったものが戻る」候補は、前日比マイナスに加えて直近15分のプラス転換・安値からの回復・VWAP付近への復帰を要求します。下落しているだけでは反発扱いにしません。<br><br>
<b>重要：</b>研究知見を測定可能な代理変数へ落とした試作であり、収益性を保証する統計モデルではありません。無料の非公式データには遅延・欠損・仕様変更があります。売買推奨ではなく、実注文前に証券会社の板・ニュース・手数料を確認してください。根拠は<a href='https://github.com/syamashi/daytrade-ranker-jp/blob/main/RESEARCH.md'>RESEARCH.md</a>に記載しています。</div>
<!-- data:{payload}; errors:{html.escape('; '.join(errors))} --></main></body></html>"""


def main() -> None:
    market_momentum, market_window = benchmark_return(fetch_chart("1306"))
    rows, errors = [], []
    for item in load_tickers():
        try:
            rows.append(metrics(item, fetch_chart(item["code"]), market_momentum))
        except Exception as exc:
            errors.append(f"{item['code']}: {exc}")
    if not rows:
        raise RuntimeError("All ticker downloads failed: " + "; ".join(errors))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        render(score_all(rows), errors, market_momentum, market_window), encoding="utf-8"
    )
    print(f"Generated {OUTPUT_FILE} ({len(rows)} tickers, {len(errors)} errors)")


if __name__ == "__main__":
    main()
