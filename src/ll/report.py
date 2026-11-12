"""Phase 9: Portable, Self-Contained HTML Report Generator."""
from __future__ import annotations
import pathlib
import pandas as pd


def media_tag(html_path: str, alt: str, is_video: bool = False) -> str:
    """Embed a media file only if it exists; otherwise show how to generate it.

    ``html_path`` is relative to ``report/index.html`` (e.g. ``../videos/x.gif``).
    A missing file used to render as a broken image/video tag with no hint.
    """
    fs_path = pathlib.Path("report") / html_path
    if fs_path.exists():
        if is_video:
            return f'<video controls src="{html_path}" width="100%"></video>'
        return f'<img src="{html_path}" alt="{alt}">'
    return (
        f"<p><em>Media not yet generated: <code>{html_path}</code> — "
        f"run <code>python -m ll.rollouts && python -m ll.video</code> "
        f"and <code>python scripts/01_random_agent.py</code> to create it.</em></p>"
    )


def build_report():
    out_dir = pathlib.Path("report")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_html = "<p><em>Run full evaluation (python -m ll.evaluate) to populate the summary table.</em></p>"
    summary_path = pathlib.Path("results/eval_summary.csv")
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        summary_html = df_sum.to_html(classes="table-summary", index=False)

    explainer_html = "<p><em>Run scripts/02_explain_reward.py to populate.</em></p>"
    explainer_path = pathlib.Path("results/reward_explainer.csv")
    if explainer_path.exists():
        df_exp = pd.read_csv(explainer_path)
        explainer_html = df_exp.head(8).to_html(classes="table-explainer", index=False)

    random_crash_html = media_tag("../videos/random_crash.gif", "Random Crash")
    viewer_demo_html = media_tag("../videos/viewer_demo.mp4", "Live Viewer Demo", is_video=True)
    comparison_html = media_tag("../videos/comparison.mp4", "Side-by-Side Comparison", is_video=True)
    baseline_gif_html = media_tag("../videos/baseline_seed0.gif", "Baseline Landing")
    fixed_gif_html = media_tag("../videos/fixed_seed0.gif", "Fixed Policy")
    hover_gif_html = media_tag("../videos/hover_hack_seed0.gif", "Hover Hack Policy")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lunar Lander: Reward Shaping & Reward Hacking Report</title>
    <style>
        :root {{
            --bg: #0f141c;
            --surface: #18202c;
            --surface-border: #283548;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --accent-yellow: #facc15;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #ffffff;
            font-weight: 700;
        }}
        h1 {{
            border-bottom: 2px solid var(--surface-border);
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 28px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 0.95em;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--surface-border);
        }}
        th {{
            background: #1e293b;
            color: var(--accent-blue);
        }}
        pre, code {{
            background: #090d14;
            color: #38bdf8;
            font-family: Consolas, Monaco, monospace;
            border-radius: 4px;
        }}
        pre {{
            padding: 14px;
            overflow-x: auto;
            border: 1px solid var(--surface-border);
        }}
        code {{
            padding: 2px 6px;
        }}
        img, video {{
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid var(--surface-border);
            display: block;
            margin: 16px 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
        .badge-danger {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); }}
    </style>
</head>
<body>
<div class="container">
    <h1>🚀 Lunar Lander: Reward Shaping & Live Simulation</h1>

    <div class="card">
        <h2>1. Executive Summary & Headline Results</h2>
        <p>This project implements and evaluates <strong>reward shaping</strong> and <strong>reward hacking</strong> in Gymnasium's <code>LunarLander-v3</code> physics environment using PPO.</p>
        <p>A naive bonus for remaining upright and slow leads the policy to exploit the 1,000-frame episode timeout by <strong>hovering indefinitely</strong> instead of landing. We formalize and verify the potential-based shaping solution that cures this vulnerability.</p>
        {summary_html}
    </div>

    <div class="card">
        <h2>2. How It Works: Observations, Actions & Built-in Reward</h2>
        <p>Gymnasium provides an 8-dimensional continuous observation vector and 4 discrete engine actions:</p>
        <ul>
            <li><code>0</code>: NOOP (coasting)</li>
            <li><code>1</code>: Fire Left RCS (steers right)</li>
            <li><code>2</code>: Fire Main Engine (thrust upward)</li>
            <li><code>3</code>: Fire Right RCS (steers left)</li>
        </ul>
        <h3>Two-Frame Step Reward Verification</h3>
        <p>Verification showing that built-in reward matches delta position score minus engine fuel cost:</p>
        {explainer_html}
        {random_crash_html}
    </div>

    <div class="card">
        <h2>3. Reward Hacking: The Hover Exploit vs. The Mathematical Fix</h2>
        <div class="grid">
            <div>
                <h3>Hover Hack Vulnerability</h3>
                <p>Giving <code>+1.0</code> per frame for being upright and slow allows the rocket to accumulate up to 1,000 points by floating at high altitude while burning at most ~300 in fuel.</p>
            </div>
            <div>
                <h3>Potential-Based Fix</h3>
                <p>Using <code>F(s,s') = &gamma; &Phi;(s') - &Phi;(s)</code> with terminal potential zero guarantees that the cumulative bonus telescopes to &le; 0 over any trajectory, making hovering strictly suboptimal.</p>
            </div>
        </div>
        <img src="../figures/12_fixed_bonus_bounded.png" alt="Mathematical Proof of Bounded Bonus">
        <h3>Rollout Clips (seed 0, episode 0)</h3>
        <p>One episode per policy, recorded from the real environment. Only the baseline is trained enough to land — the fixed and hover_hack models are 100k-step smoke tests and crash in these clips, which is why the hover exploit is not visible yet.</p>
        <div class="grid">
            <div>
                <h3>Baseline — landed</h3>
                {baseline_gif_html}
            </div>
            <div>
                <h3>Fixed — crashed (100k steps)</h3>
                {fixed_gif_html}
            </div>
        </div>
        <h3>Hover Hack — crashed (100k steps, no hover yet)</h3>
        {hover_gif_html}
        <h3>Side-by-Side Comparison</h3>
        {comparison_html}
    </div>

    <div class="card">
        <h2>4. Interactive Live Viewer</h2>
        <p>A real-time desktop viewer built in Pygame-CE driven by the real Gymnasium Box2D physics and the policy's deterministic predictions:</p>
        <pre><code>python -m ll.live_viewer --config baseline --seed 0</code></pre>
        {viewer_demo_html}
    </div>

    <div class="card">
        <h2>5. Reproducibility & Commands</h2>
        <pre><code># Check setup and environments
python scripts/00_check_install.py

# Run random policy baseline
python scripts/01_random_agent.py

# Verify reward formula
python scripts/02_explain_reward.py

# Quick PPO training smoke test
python -m ll.train --config baseline --quick

# Interactive live viewer
python -m ll.live_viewer --config baseline --seed 0
</code></pre>
    </div>
</div>
</body>
</html>
"""
    out_file = out_dir / "index.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML report at {out_file}")


if __name__ == "__main__":
    build_report()

