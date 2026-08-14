from __future__ import annotations

import csv
import html
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
TICKERS_FILE = ROOT / "tickers.csv"
OUTPUT_FILE = ROOT / "docs" / "index.html"
MODEL_FILE = ROOT / "docs" / "simulation.json"
ALERT_FILE = ROOT / "docs" / "alert.json"
JST = ZoneInfo("Asia/Tokyo")
PAPER_CAPITAL_YEN = 1_000_000
MAX_POSITION_YEN = 300_000
MIN_EXPECTED_NET_PCT = 0.10


def load_tickers() -> list[dict[str, str]]:
    with TICKERS_FILE.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fetch_chart(code: str, data_range: str = "5d") -> dict:
    symbol = f"{code}.T"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    query = urllib.parse.urlencode(
        {"interval": "5m", "range": data_range, "includePrePost": "false"}
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


def score_all(rows: list[dict], model: dict | None = None) -> list[dict]:
    fields = [
        "relative_strength", "momentum_15m", "relative_volume", "range_position",
        "efficiency", "flip_rate", "turnover_million", "realized_vol",
        "day_change", "recovery_from_low", "vwap_gap",
    ]
    ranks = {field: percentiles([r[field] for r in rows]) for field in fields}
    ranks["near_vwap"] = percentiles([-abs(r["vwap_gap"]) for r in rows])

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
        model_features = {
            "relative_strength": p["relative_strength"],
            "momentum_15m": p["momentum_15m"],
            "relative_volume": p["relative_volume"],
            "range_position": p["range_position"],
            "efficiency": p["efficiency"],
            "low_flip": low_chop,
            "turnover": p["turnover_million"],
            "volatility_sweet": volatility_sweet_spot,
            "negative_day": 1 - p["day_change"],
            "recovery": p["recovery_from_low"],
            "near_vwap": ranks["near_vwap"][i],
        }
        weights = model.get("weights", {}) if model else {}
        row["model_score"] = round(
            sum(float(weights.get(name, 0.0)) * value for name, value in model_features.items()),
            4,
        )
        if best < 55:
            row["signal"] = "様子見"
        elif trend >= rebound:
            row["signal"] = "上昇継続"
        else:
            row["signal"] = "反発"
    sort_key = "model_score" if model else "score"
    return sorted(rows, key=lambda x: x[sort_key], reverse=True)


def load_model() -> dict | None:
    if not MODEL_FILE.exists():
        return None
    try:
        return json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def market_entry_window(now: datetime) -> tuple[bool, str]:
    if now.weekday() >= 5:
        return False, "次の営業日09:00以降に再判定"
    minute = now.hour * 60 + now.minute
    if 9 * 60 <= minute <= 11 * 60 + 20 or 12 * 60 + 30 <= minute <= 15 * 60 + 15:
        end = now + timedelta(minutes=15)
        return True, f"{now:%H:%M}〜{end:%H:%M}"
    if minute < 9 * 60:
        return False, "09:00以降に再判定"
    if minute < 12 * 60 + 30:
        return False, "12:30以降に再判定"
    return False, "翌営業日09:00以降に再判定"


def build_recommendation(
    rows: list[dict], model: dict | None, now: datetime | None = None
) -> dict:
    now = now or datetime.now(JST)
    market_open, time_window = market_entry_window(now)
    expected_net = float((model or {}).get("validation", {}).get("avg_net_pct", 0.0))
    threshold = float((model or {}).get("threshold", 1.0))
    affordable = [r for r in rows if r["price"] * 100 <= MAX_POSITION_YEN]
    candidate = affordable[0] if affordable else (rows[0] if rows else None)
    reasons = []
    if not model:
        reasons.append("最適化モデルがまだありません")
    elif not model.get("paper_ready"):
        reasons.append("学習・検証・未使用テストの合格条件を満たしていません")
    if expected_net < MIN_EXPECTED_NET_PCT:
        reasons.append(
            f"コスト控除後の検証期待値 {expected_net:+.3f}% が通知閾値 "
            f"{MIN_EXPECTED_NET_PCT:.2f}% 未満です"
        )
    if not market_open:
        reasons.append("東証の連続取引時間外です")
    if candidate is None:
        reasons.append("取得できた銘柄がありません")
    elif candidate["price"] * 100 > MAX_POSITION_YEN:
        reasons.append("100株の参考購入額が仮想運用の上限を超えます")
    elif candidate["model_score"] < threshold:
        reasons.append(
            f"モデル評価値 {candidate['model_score']:.3f} が採用閾値 {threshold:.3f} 未満です"
        )
    elif candidate["window_minutes"] < 30:
        reasons.append("直近データが30分未満です")

    announce = not reasons
    if candidate:
        shares = 100
        amount = round(candidate["price"] * shares)
        stop_pct = min(1.5, max(0.7, candidate["realized_vol"] * 1.5))
        stop_price = candidate["price"] * (1 - stop_pct / 100)
        candidate_data = {
            "code": candidate["code"], "name": candidate["name"],
            "price": round(candidate["price"], 1), "shares": shares,
            "amount_yen": amount, "model_score": candidate["model_score"],
            "model_threshold": threshold, "expected_net_pct": expected_net,
            "reference_stop_price": round(stop_price, 1),
            "reference_stop_pct": round(stop_pct, 2), "time_window": time_window,
        }
    else:
        candidate_data = None

    title = (
        f"【仮想注文候補】{candidate['code']} {candidate['name']} {now:%m/%d %H:%M}"
        if announce and candidate else f"取引なし {now:%m/%d %H:%M}"
    )
    if announce and candidate_data:
        body = (
            f"銘柄: {candidate_data['code']} {candidate_data['name']}\n\n"
            f"参考注文: {candidate_data['shares']}株 / 約{candidate_data['amount_yen']:,}円\n\n"
            f"確認時間: {candidate_data['time_window']}\n\n"
            f"参考損切り: {candidate_data['reference_stop_price']:,.1f}円 "
            f"(-{candidate_data['reference_stop_pct']:.2f}%)\n\n"
            f"検証期待値: {expected_net:+.3f}%（売買コスト控除後の検証期間平均）\n\n"
            "これは自動売買や利益保証ではなく、100万円の仮想運用候補です。発注はしません。"
        )
    else:
        body = "通知条件を満たしていません。\n\n- " + "\n- ".join(reasons)
    return {
        "announce": announce, "title": title, "body": body,
        "generated_at": now.strftime("%Y-%m-%d %H:%M JST"),
        "reasons": reasons, "candidate": candidate_data,
        "policy": {
            "paper_capital_yen": PAPER_CAPITAL_YEN,
            "max_position_yen": MAX_POSITION_YEN,
            "min_expected_net_pct": MIN_EXPECTED_NET_PCT,
            "board_lot_shares": 100,
        },
    }


def render(
    rows: list[dict], errors: list[str], market_momentum: float, market_window: int,
    recommendation: dict,
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
    candidate = recommendation.get("candidate")
    if recommendation["announce"] and candidate:
        recommendation_html = f"""<section class='order ready'><h2>仮想注文候補</h2>
<div class='order-grid'><b>{html.escape(candidate['code'])} {html.escape(candidate['name'])}</b>
<span>参考注文：{candidate['shares']}株 / 約{candidate['amount_yen']:,}円</span>
<span>確認時間：{html.escape(candidate['time_window'])}</span>
<span>参考損切り：{candidate['reference_stop_price']:,.1f}円（-{candidate['reference_stop_pct']:.2f}%）</span>
<span>検証期待値：{candidate['expected_net_pct']:+.3f}%</span></div>
<p>売買コスト控除後の検証期間平均です。自動発注はせず、100万円の仮想運用として記録してください。</p></section>"""
    else:
        reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in recommendation["reasons"])
        if candidate:
            candidate_summary = f"""<div class='order-grid'><b>最上位（条件未達）：{html.escape(candidate['code'])} {html.escape(candidate['name'])}</b>
<span>参考注文：{candidate['shares']}株 / 約{candidate['amount_yen']:,}円</span>
<span>確認時間：{html.escape(candidate['time_window'])}</span>
<span>検証期待値：{candidate['expected_net_pct']:+.3f}%</span></div>"""
        else:
            candidate_summary = ""
        recommendation_html = f"""<section class='order wait'><h2>今回の判定：取引なし</h2>
{candidate_summary}<ul>{reasons}</ul><p>最上位候補は表示しますが、条件が全部そろうまで通知も発注もしません。</p></section>"""
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
.notice{{margin-top:16px;padding:14px;background:#17202b;border-radius:10px;font-size:13px;line-height:1.65}}.order{{margin:16px 0;padding:18px;border-radius:14px;border:2px solid}}.order.ready{{background:#12372d;border-color:#54d6a0}}.order.wait{{background:#30271e;border-color:#d99b52}}.order h2{{margin:0 0 10px}}.order-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}}.order-grid>*{{padding:10px;background:#0b1725;border-radius:8px}}.order ul{{margin:8px 0}}a{{color:#7ab8ff}}@media(max-width:600px){{main{{padding:12px}}}}
</style></head><body><main><h1>日本株デイトレ候補ランキング</h1><p>最終更新：{now}　データ時点：{html.escape(rows[0]['asof']) if rows else '-'}　監視：{len(rows)}銘柄</p>
{recommendation_html}
<div class='summary'><span class='pill'>市場直近{market_window}分：{market_momentum:+.2f}%</span><span class='pill'>総合＝シグナル強度×売買しやすさ</span><span class='pill'>上昇継続と反発を別採点</span><a class='pill' href='simulation.html'>過去60日の仮想運用を見る</a></div>
<div class='card'><table><thead><tr><th>#</th><th>銘柄</th><th>判定</th><th>総合</th><th>継続</th><th>反発</th><th>現在値</th><th>前日比</th><th>市場超過・直近</th><th>15分</th><th>出来高比</th><th>値幅率</th><th>レンジ位置</th><th>価格効率</th><th>反転頻度</th><th>VWAP乖離</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>評価に使う要素</h2><div class='factors'>{factor_cards}</div>
<div class='notice'><b>読み方：</b>「上がったものが上がり続ける」候補は、相対強度・出来高・高値圏維持・低い反転頻度を重視します。「今日下がったものが戻る」候補は、前日比マイナスに加えて直近15分のプラス転換・安値からの回復・VWAP付近への復帰を要求します。下落しているだけでは反発扱いにしません。<br><br>
<b>重要：</b>研究知見を測定可能な代理変数へ落とした試作であり、収益性を保証する統計モデルではありません。無料の非公式データには遅延・欠損・仕様変更があります。売買推奨ではなく、実注文前に証券会社の板・ニュース・手数料を確認してください。根拠は<a href='https://github.com/syamashi/daytrade-ranker-jp/blob/main/RESEARCH.md'>RESEARCH.md</a>に記載しています。</div>
<!-- data:{payload}; errors:{html.escape('; '.join(errors))} --></main></body></html>"""


def main() -> None:
    model = load_model()
    market_momentum, market_window = benchmark_return(fetch_chart("1306"))
    rows, errors = [], []
    for item in load_tickers():
        try:
            rows.append(metrics(item, fetch_chart(item["code"]), market_momentum))
        except Exception as exc:
            errors.append(f"{item['code']}: {exc}")
    if not rows:
        raise RuntimeError("All ticker downloads failed: " + "; ".join(errors))
    ranked = score_all(rows, model)
    recommendation = build_recommendation(ranked, model)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_FILE.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_FILE.write_text(
        render(ranked, errors, market_momentum, market_window, recommendation),
        encoding="utf-8",
    )
    print(
        f"Generated {OUTPUT_FILE} ({len(rows)} tickers, {len(errors)} errors, "
        f"announce={recommendation['announce']})"
    )


if __name__ == "__main__":
    main()
