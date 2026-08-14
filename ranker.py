from __future__ import annotations

import csv
import html
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
TICKERS_FILE = ROOT / "tickers.csv"
OUTPUT_FILE = ROOT / "docs" / "index.html"


def load_tickers() -> list[dict[str, str]]:
    with TICKERS_FILE.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fetch_chart(code: str) -> dict:
    symbol = f"{code}.T"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    query = urllib.parse.urlencode({"interval": "5m", "range": "5d", "includePrePost": "false"})
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "Mozilla/5.0 daytrade-ranker/0.1"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)["chart"]["result"]
    if not result:
        raise ValueError(f"No data: {symbol}")
    return result[0]


def safe_z(value: float, values: list[float]) -> float:
    sigma = pstdev(values) if len(values) > 1 else 0
    return (value - mean(values)) / sigma if sigma > 1e-12 else 0.0


def metrics(item: dict[str, str], chart: dict) -> dict:
    timestamps = chart["timestamp"]
    quote = chart["indicators"]["quote"][0]
    rows = []
    for ts, close, high, low, volume in zip(
        timestamps, quote["close"], quote["high"], quote["low"], quote["volume"]
    ):
        if None not in (close, high, low, volume):
            rows.append((ts, float(close), float(high), float(low), float(volume)))
    if len(rows) < 20:
        raise ValueError("Not enough intraday data")

    days: dict[str, list[tuple]] = {}
    for row in rows:
        day = datetime.fromtimestamp(row[0], ZoneInfo("Asia/Tokyo")).date().isoformat()
        days.setdefault(day, []).append(row)
    today = days[sorted(days)[-1]]
    previous = [r for day in sorted(days)[:-1] for r in days[day]]
    if len(today) < 3 or not previous:
        raise ValueError("Not enough comparable data")

    close = today[-1][1]
    open_price = today[0][1]
    prev_close = previous[-1][1]
    momentum_1h = (close / today[max(0, len(today) - 13)][1] - 1) * 100
    day_change = (close / prev_close - 1) * 100
    typical_value = sum(((h + l + c) / 3) * v for _, c, h, l, v in today)
    volume_today = sum(r[4] for r in today)
    vwap = typical_value / volume_today if volume_today else close
    vwap_gap = (close / vwap - 1) * 100

    slot = min(len(today), 12)
    current_volume = sum(r[4] for r in today[-slot:])
    historical_slots = []
    for day in sorted(days)[:-1]:
        d = days[day]
        if len(d) >= slot:
            historical_slots.append(sum(r[4] for r in d[-slot:]))
    baseline = mean(historical_slots) if historical_slots else max(current_volume, 1)
    volume_ratio = current_volume / baseline if baseline else 1
    turnover_million = close * volume_today / 1_000_000

    return {
        "code": item["code"], "name": item["name"], "price": close,
        "day_change": day_change, "momentum_1h": momentum_1h,
        "vwap_gap": vwap_gap, "volume_ratio": volume_ratio,
        "turnover_million": turnover_million, "open_change": (close / open_price - 1) * 100,
    }


def score_all(rows: list[dict]) -> list[dict]:
    fields = ["momentum_1h", "volume_ratio", "vwap_gap", "turnover_million"]
    values = {field: [math.log1p(r[field]) if field == "turnover_million" else r[field] for r in rows] for field in fields}
    for row in rows:
        raw = (
            30 * safe_z(row["momentum_1h"], values["momentum_1h"])
            + 30 * safe_z(row["volume_ratio"], values["volume_ratio"])
            + 20 * safe_z(row["vwap_gap"], values["vwap_gap"])
            + 20 * safe_z(math.log1p(row["turnover_million"]), values["turnover_million"])
        )
        overheat = max(0, abs(row["day_change"]) - 8) * 5
        row["score"] = round(max(0, min(100, 50 + raw / 3 - overheat)), 1)
    return sorted(rows, key=lambda x: x["score"], reverse=True)


def render(rows: list[dict], errors: list[str]) -> str:
    now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")
    table = "".join(
        f"<tr><td>{i}</td><td><b>{html.escape(r['code'])}</b><br><small>{html.escape(r['name'])}</small></td>"
        f"<td class='score'>{r['score']:.1f}</td><td>{r['price']:,.1f}</td>"
        f"<td class='{'up' if r['day_change'] >= 0 else 'down'}'>{r['day_change']:+.2f}%</td>"
        f"<td>{r['momentum_1h']:+.2f}%</td><td>{r['volume_ratio']:.2f}x</td>"
        f"<td>{r['vwap_gap']:+.2f}%</td><td>{r['turnover_million']:,.0f}</td></tr>"
        for i, r in enumerate(rows, 1)
    )
    payload = html.escape(json.dumps(rows, ensure_ascii=False))
    return f"""<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>デイトレ候補ランキング</title><style>
:root{{--bg:#07111f;--panel:#101d2d;--text:#e8f0f7;--muted:#91a4b7;--accent:#54d6a0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:20px}}h1{{font-size:clamp(22px,4vw,34px);margin:0 0 5px}}p{{color:var(--muted)}}
.card{{background:var(--panel);border:1px solid #20354b;border-radius:14px;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:900px}}
th,td{{padding:11px 12px;border-bottom:1px solid #20354b;text-align:right}}th{{position:sticky;top:0;background:#15263a;color:#b9c8d6}}
th:nth-child(2),td:nth-child(2){{text-align:left}}.score{{font-size:18px;color:var(--accent);font-weight:800}}.up{{color:#ff6f7d}}.down{{color:#65aaff}}small{{color:var(--muted)}}
.notice{{margin-top:16px;padding:14px;background:#17202b;border-radius:10px;font-size:13px}}@media(max-width:600px){{main{{padding:12px}}}}
</style></head><body><main><h1>デイトレ候補ランキング</h1><p>最終更新：{now}　監視銘柄：{len(rows)}件</p>
<div class='card'><table><thead><tr><th>#</th><th>銘柄</th><th>評価</th><th>現在値</th><th>前日比</th><th>1時間</th><th>出来高比</th><th>VWAP乖離</th><th>売買代金(百万円)</th></tr></thead><tbody>{table}</tbody></table></div>
<div class='notice'>これは投資判断の補助を目的とした試作品です。売買推奨ではありません。無料の非公式データを使うため、遅延・欠損・仕様変更の可能性があります。実売買前に必ず証券会社の画面で確認してください。</div>
<!-- data:{payload}; errors:{html.escape('; '.join(errors))} --></main></body></html>"""


def main() -> None:
    rows, errors = [], []
    for item in load_tickers():
        try:
            rows.append(metrics(item, fetch_chart(item["code"])))
        except Exception as exc:
            errors.append(f"{item['code']}: {exc}")
    if not rows:
        raise RuntimeError("All ticker downloads failed: " + "; ".join(errors))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(render(score_all(rows), errors), encoding="utf-8")
    print(f"Generated {OUTPUT_FILE} ({len(rows)} tickers, {len(errors)} errors)")


if __name__ == "__main__":
    main()
