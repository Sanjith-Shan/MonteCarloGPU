#!/usr/bin/env python3
"""Build a self contained HTML performance dashboard.

Reads the JSON result files plus the ML reports and renders a single HTML page
with the headline numbers, a benchmark table, and the charts embedded as base64
so the file is portable with no external assets. This is the reporting layer
that turns raw benchmark output into something a reviewer can read at a glance.

Usage:
    python3 dashboard/generate_dashboard.py
    # writes dashboard/dashboard.html
"""

from __future__ import annotations

import base64
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
ML = os.path.join(ROOT, "ml")
OUT = os.path.join(os.path.dirname(__file__), "dashboard.html")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def embed_png(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def stat_card(label, value, sub=""):
    return f"""<div class="card"><div class="value">{value}</div>
    <div class="label">{label}</div><div class="sub">{sub}</div></div>"""


def chart_block(title, data_uri):
    if not data_uri:
        return ""
    return f"""<div class="chart"><h3>{title}</h3>
    <img src="{data_uri}" alt="{title}"/></div>"""


def build():
    bench = load_json(os.path.join(RESULTS, "benchmark.json"))
    conv = load_json(os.path.join(RESULTS, "convergence.json"))
    ml_metrics = load_json(os.path.join(ML, "metrics.json"))
    ml_infer = load_json(os.path.join(ML, "infer_report.json"))

    device = bench["meta"]["device"] if bench else "n/a"
    peak_speedup = 0.0
    bench_rows = ""
    if bench:
        for r in bench["scaling"]:
            sp = r["speedup"]
            if sp:
                peak_speedup = max(peak_speedup, sp)
            cpu = f"{r['cpu_ms']:.1f}" if r["cpu_ms"] else "n/a"
            sps = f"{sp:.1f}x" if sp else "n/a"
            bench_rows += (f"<tr><td>{r['paths']:,}</td><td>{cpu}</td>"
                           f"<td>{r['gpu_ms']:.1f}</td><td>{sps}</td>"
                           f"<td>{r['gpu_price']:.4f}</td></tr>")

    cards = stat_card("Peak GPU speedup", f"{peak_speedup:.0f}x", f"on {device}")
    if conv:
        cards += stat_card("Convergence slope", f"{conv['slope']:.3f}",
                           "theory -0.500")
    if ml_infer:
        cards += stat_card("Surrogate throughput",
                           f"{ml_infer['throughput_per_s']/1e3:.0f}k/s",
                           f"{ml_infer['per_option_us']:.2f} us per option")
    if ml_metrics:
        cards += stat_card("Surrogate val RMSE",
                           f"{ml_metrics['best_val_rmse']:.4f}",
                           "per unit strike")

    charts = ""
    charts += chart_block("CPU vs GPU pricing time",
                          embed_png(os.path.join(RESULTS, "speedup.png")))
    charts += chart_block("Convergence to Black Scholes",
                          embed_png(os.path.join(RESULTS, "convergence.png")))
    charts += chart_block("Greeks, Monte Carlo vs analytical",
                          embed_png(os.path.join(RESULTS, "greeks.png")))
    charts += chart_block("Price surface",
                          embed_png(os.path.join(RESULTS, "surface.png")))
    charts += chart_block("Surrogate training curve",
                          embed_png(os.path.join(ML, "loss_curve.png")))

    html = TEMPLATE.format(device=device, cards=cards, rows=bench_rows,
                           charts=charts)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT}")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MonteCarloGPU Performance Dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 0; background: #0f1115; color: #e8e8e6; }}
  header {{ padding: 28px 32px; border-bottom: 1px solid #262a33; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header p {{ margin: 6px 0 0; color: #9aa0ab; font-size: 14px; }}
  .accent {{ color: #76b900; }}
  main {{ padding: 28px 32px; max-width: 1100px; margin: 0 auto; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
           gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #171a21; border: 1px solid #262a33; border-radius: 12px;
          padding: 20px; }}
  .card .value {{ font-size: 30px; font-weight: 700; color: #76b900; }}
  .card .label {{ margin-top: 4px; font-size: 14px; }}
  .card .sub {{ color: #9aa0ab; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px;
          font-size: 14px; }}
  th, td {{ text-align: right; padding: 8px 12px; border-bottom: 1px solid #262a33; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: #9aa0ab; font-weight: 600; }}
  .chart {{ background: #fff; border-radius: 12px; padding: 12px; margin-bottom: 20px; }}
  .chart h3 {{ color: #23262c; margin: 4px 8px 10px; font-size: 15px; }}
  .chart img {{ width: 100%; height: auto; border-radius: 6px; }}
</style>
</head>
<body>
<header>
  <h1>MonteCarlo<span class="accent">GPU</span> Performance Dashboard</h1>
  <p>GPU accelerated Monte Carlo option pricing. Device {device}.</p>
</header>
<main>
  <div class="cards">{cards}</div>
  <h2>Scaling benchmark</h2>
  <table>
    <thead><tr><th>Paths</th><th>CPU ms</th><th>GPU ms</th>
    <th>Speedup</th><th>Price</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Charts</h2>
  {charts}
</main>
</body>
</html>
"""


if __name__ == "__main__":
    build()
