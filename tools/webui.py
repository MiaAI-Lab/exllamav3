#!/usr/bin/env python3
"""
Lightweight web UI for launching EXL3 conversions on this box.

Run:
  python tools/webui.py [--port 7860] [--host 127.0.0.1]
then open http://127.0.0.1:7860 in a browser.

Feeds tools/quantize_model.py (detached) and tails its log. Nothing here is
CPU/GPU heavy; all conversions run as background subprocesses.
"""
import argparse, glob, json, os, subprocess, sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TOOLS = os.path.join(REPO, "tools")
JOBS = os.path.join(REPO, "jobs.jsonl")
SOURCES = os.path.join(REPO, "test_models", "sources")

import gradio as gr


def recent_sources():
    names = [p for p in sorted(glob.glob(os.path.join(SOURCES, "*"))) if os.path.isdir(p)]
    models = []
    for p in names:
        if glob.glob(os.path.join(p, "*.safetensors")) or glob.glob(os.path.join(p, "*.bin")):
            models.append(os.path.basename(p))
    jobs = []
    if os.path.exists(JOBS):
        for line in open(JOBS):
            try:
                r = json.loads(line)
                if r.get("out"):
                    jobs.append(os.path.basename(r["out"]))
            except Exception:
                pass
    return sorted(set(models)), sorted(set(jobs))


def sysinfo():
    out = []
    for cmd in (["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                ["free", "-h"],
                ["df", "-h", REPO]):
        try:
            out.append("$ " + " ".join(cmd) + "\n" + subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip())
        except Exception as e:
            out.append(f"(err) {' '.join(cmd)}: {e}")
    disk = subprocess.run(["df", "-h", REPO], capture_output=True, text=True).stdout.strip()
    return "\n\n".join(out) + ("\n\nWORKSPACE DISK\n" + disk)


def run_preflight(model, bits, hq):
    cmd = [sys.executable, os.path.join(TOOLS, "quantize_model.py"), "-m", model, "-b", str(bits)] + (["--hq"] if hq else []) + ["--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=1800)
    return (r.stdout + r.stderr)[-8000:]


def start_job(model, bits, hq, name, draft):
    cmd = [sys.executable, os.path.join(TOOLS, "quantize_model.py"), "-m", model, "-b", str(bits),
           "--detach", "--name", (name or model.split('/')[-1])]
    if hq:
        cmd.append("--hq")
    if draft:
        cmd += ["--draft-model-dir", draft]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=120)
    return (r.stdout + r.stderr)[-4000:]


def jobs_table():
    rows = []
    if os.path.exists(JOBS):
        for line in open(JOBS):
            try:
                r = json.loads(line)
                rows.append([r.get("time", ""), r.get("name", ""), r.get("bits", ""), r.get("status", ""),
                             r.get("out", ""), str(r.get("elapsed_s", ""))])
            except Exception:
                continue
    return rows


def live_log():
    last_log = None
    if os.path.exists(JOBS):
        for line in open(JOBS):
            try:
                r = json.loads(line)
                last_log = r.get("log")
            except Exception:
                pass
    body = ""
    if last_log and os.path.exists(last_log):
        body = open(last_log, errors="replace").read()[-6000:]
    return (last_log or ""), body


with gr.Blocks(title="EXL3 Quantizer") as demo:
    gr.Markdown("# EXL3 model quantizer — DGX Spark")
    with gr.Tab("Convert"):
        with gr.Row():
            src = gr.Textbox(label="Source: local dir or HF repo id", placeholder="Qwen/Qwen3.8-27B")
            name = gr.Textbox(label="Job name (optional)")
        with gr.Row():
            bits = gr.Radio(["3.0", "3.5", "4.0"], value="3.5", label="Bits per weight")
            hq = gr.Checkbox(value=True, label="-hq (protect attention)")
        with gr.Row():
            draft = gr.Textbox(label="Speculative draft model dir (optional, packaged into output/draft)",
                               placeholder="test_models/sources/Qwen3.8-27B-DSpark")
        with gr.Row():
            pre = gr.Button("Preflight only (download + check support)")
            start = gr.Button("Start conversion", variant="primary")
        preout = gr.Textbox(label="Preflight output", lines=8)
        startout = gr.Textbox(label="Launch output", lines=4)
        gr.Markdown("### Live log")
        logbox = gr.Textbox(label="Active job log tail (auto-refresh)", lines=20)
        logpath = gr.Textbox(label="Log file", interactive=False)
        refresh = gr.Button("Refresh log")
        pre.click(run_preflight, [src, bits, hq], preout)
        start.click(start_job, [src, bits, hq, name, draft], startout)
        refresh.click(live_log, outputs=[logpath, logbox])
        demo.load(live_log, outputs=[logpath, logbox])
    with gr.Tab("Jobs"):
        grid = gr.Dataframe(headers=["time", "name", "bits", "status", "out", "elapsed_s"], interactive=False)
        jref = gr.Button("Refresh jobs")
        jref.click(jobs_table, outputs=grid)
        demo.load(jobs_table, outputs=grid)
    with gr.Tab("System"):
        sinfo = gr.Textbox(label="GPU / RAM / disk", lines=20)
        gr.Button("Refresh system info").click(sysinfo, outputs=sinfo)
        demo.load(sysinfo, outputs=sinfo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue(default_concurrency_limit=4).launch(server_name=a.host, server_port=a.port)
