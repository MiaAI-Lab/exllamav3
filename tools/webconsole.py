#!/usr/bin/env python3
"""
EXL3 web console — a complete web UI over this repo's CLI surface.

Covers everything the CLI can do, as thin wrappers over the same scripts:
  * Quantize      tools/quantize_model.py + convert.py (start / preflight / resume / attach draft)
  * Serve         tools/serve_openai.py (start / stop / health)
  * Evaluate      eval/*.py
  * Tools         tools/*.py (dequant, benches, probes)
  * Chat          playground for a running serve_openai endpoint (SSE streaming)
  * Models        browser for test_models/ and test_models/sources/
  * Jobs          unified job registry (runs/webui/jobs.jsonl) + logs + kill
  * System        GPU / RAM / disk / versions

Run:
  .venv/bin/python tools/webconsole.py [--host 127.0.0.1] [--port 7861]

The console itself is light; every heavy operation runs as a detached
subprocess of the repo's own CLI scripts (new session, survives a console
restart, killable from the UI). No dependencies beyond what serve_openai.py
already uses (aiohttp). The legacy Gradio launcher (tools/webui.py) stays.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))          # tools/
PKG = os.path.join(HERE, "webconsole")                     # tools/webconsole/
REPO = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, REPO)

from aiohttp import web, ClientSession, ClientTimeout  # noqa: E402
from jobman import JobManager  # noqa: E402
import specs as S  # noqa: E402

PY = sys.executable
JOBS = JobManager(REPO)
STATIC = os.path.join(PKG, "static")
TEST_MODELS = os.path.join(REPO, "test_models")
LEGACY_JOBS = os.path.join(REPO, "jobs.jsonl")


def err(msg, status=400):
    return web.json_response({"error": msg}, status=status)


# --------------------------------------------------------------------- helpers

def spec_by_key(key):
    for s in S.SPECS:
        if s["key"] == key:
            return s
    return None


def clean_params(spec, params):
    """Validate/coerce client params against the spec's fields."""
    out = {}
    for fld in spec["fields"]:
        name = fld["name"]
        val = params.get(name)
        if val in (None, ""):
            val = fld.get("default")
        if val in (None, ""):
            if fld.get("required"):
                raise ValueError(f"'{fld['label']}' is required")
            continue
        kind = fld["kind"]
        try:
            if kind == "int":
                val = int(val)
            elif kind == "float":
                val = float(val)
            elif kind == "bool":
                val = bool(val) if isinstance(val, bool) else str(val).lower() in ("1", "true", "on")
            else:
                val = str(val).strip()
        except (TypeError, ValueError):
            raise ValueError(f"'{fld['label']}' must be a {kind}")
        if kind == "choice" and fld.get("choices") and str(val) not in fld["choices"]:
            raise ValueError(f"'{fld['label']}' must be one of {fld['choices']}")
        out[name] = val
    return out


def generic_argv(spec, p):
    """Translate cleaned params into script flags from the field definitions."""
    argv = [PY, os.path.join(REPO, spec["script"])]
    positional = []
    for fld in spec["fields"]:
        name = fld["name"]
        if fld["kind"] == "env" or name not in p:
            continue
        val = p[name]
        if fld.get("positional"):
            positional.append(str(val))
            continue
        flag = fld.get("flag", name)
        if fld["kind"] == "bool":
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    argv += positional
    return argv


def generic_env(spec, p):
    """Collect env-kind fields into an environment overlay."""
    env = {}
    for fld in spec["fields"]:
        if fld["kind"] == "env" and p.get(fld["name"]) not in (None, ""):
            env[fld["flag"]] = str(p[fld["name"]])
    return env


def build_job(spec, p):
    """spec + cleaned params -> (kind, name, argv, meta, env_extra)"""
    key = spec["key"]
    qm = os.path.join(REPO, "tools", "quantize_model.py")

    if key == "quant":
        name = p.get("name") or os.path.basename(str(p["model"]).rstrip("/"))

        def build(log_path):
            convert_log = log_path + ".convert"
            argv = [PY, qm, "-m", p["model"], "-b", p["bits"], "--name", name,
                    "--log", convert_log]
            if p.get("hq"):
                argv.append("--hq")
            for src, flag in (("out", "--out"), ("work", "--work"), ("cal_rows", "--cr"),
                              ("cal_cols", "--cc"), ("cal_data", "--cal-data"),
                              ("head_bits", "--hb"), ("shard_size", "--ss"),
                              ("device", "-d"), ("draft_model_dir", "--draft-model-dir"),
                              ("checkpoint_interval", "--cpi")):
                if p.get(src) not in (None, ""):
                    argv += [flag, str(p[src])]
            return argv, [convert_log]

        meta = {"bits": p["bits"], "hq": bool(p.get("hq")), "model": p["model"]}
        return "quant", name, build, meta, S.QUANT_ENV

    if key == "quant_preflight":
        argv = [PY, qm, "-m", p["model"], "--dry-run"]
        return "preflight", f"preflight {os.path.basename(str(p['model']).rstrip('/'))}", \
            argv, {"model": p["model"]}, S.QUANT_ENV

    if key == "quant_resume":
        argv = [PY, qm, "-w", p["work"], "--resume"]
        if p.get("draft_model_dir"):
            argv += ["--draft-model-dir", p["draft_model_dir"]]
            if p.get("out"):
                argv += ["-o", p["out"]]
        return "quant", f"resume {os.path.basename(str(p['work']).rstrip('/'))}", \
            argv, {"work": p["work"], "resume": True}, S.QUANT_ENV

    if key == "attach_draft":
        argv = [PY, qm, "--attach-draft", "-o", p["out"],
                "--draft-model-dir", p["draft_model_dir"]]
        return "tool", f"attach-draft {os.path.basename(str(p['out']).rstrip('/'))}", \
            argv, {}, None

    if key == "serve":
        port = int(p.get("port") or 8888)
        argv = [PY, os.path.join(REPO, "tools", "serve_openai.py"),
                "-m", p["model"], "-p", str(port), "--host", p.get("host", "0.0.0.0")]
        if str(p.get("draft_model", "")).lower() not in ("", "none"):
            argv += ["-dm", p["draft_model"]]
        else:
            argv += ["-dm", "none"]
        for src, flag in (("grid_size", "-gs"), ("cache_size", "-cs"),
                          ("cache_quant", "-cq"), ("cpu_cache_size", "-ccs")):
            if p.get(src) not in (None, ""):
                argv += [flag, str(p[src])]
        env = {}
        if p.get("dspark_conf"):
            env["EXL3_DSPARK_CONF"] = str(p["dspark_conf"])
        return "serve", f"serve :{port} {os.path.basename(str(p['model']).rstrip('/'))}", \
            argv, {"port": port, "host": p.get("host", "0.0.0.0"), "model": p["model"]}, env

    if "script" in spec:
        name = spec["label"]
        return "tool" if spec["group"] == "tools" else "eval", name, \
            generic_argv(spec, p), {}, generic_env(spec, p) or None

    raise ValueError(f"no argv builder for spec '{key}'")


# --------------------------------------------------------------------- handlers

async def api_specs(_request):
    return web.json_response({"specs": S.SPECS})


async def api_system(_request):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:
            return f"(unavailable: {e})"

    version = ""
    try:
        with open(os.path.join(REPO, "exllamav3", "version.py")) as fp:
            for line in fp:
                if line.startswith("__version__"):
                    version = line.split("=", 1)[1].strip().strip('"\'')
    except OSError:
        pass

    gpu_q = sh(["nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"])
    gpus = []

    def num(x):
        try:
            return int(float(x.strip()))
        except (TypeError, ValueError):
            return None  # GB10 unified memory reports [N/A] for some fields

    for line in gpu_q.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 5:
            gpus.append({"name": parts[0], "mem_used": num(parts[1]),
                         "mem_total": num(parts[2]), "util": num(parts[3]),
                         "temp": num(parts[4])})
    mem = {}
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                k, v = line.split(":", 1)
                mem[k] = int(v.strip().split()[0]) // 1024  # kB -> MB
    except OSError:
        pass
    disk = sh(["df", "-h", "--output=target,size,used,avail,pcent", REPO])
    return web.json_response({
        "exllamav3": version,
        "python": sys.version.split()[0],
        "repo": REPO,
        "gpus": gpus,
        "ram": {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable")},
        "disk": disk,
        "load": os.getloadavg(),
    })


def _model_entry(path, kind_hint=None):
    name = os.path.basename(path.rstrip("/"))
    try:
        entries = list(os.scandir(path))
    except OSError:
        return None
    size = sum(e.stat().st_size for e in entries if e.is_file())
    names = {e.name for e in entries}
    has_config = "config.json" in names
    has_quant = "quantization_config.json" in names
    if has_config and not has_quant:
        try:
            with open(os.path.join(path, "config.json")) as fp:
                has_quant = "quantization_config" in fp.read()
        except OSError:
            pass
    has_draft = "dspark.json" in names or "draft" in names or "draft-dflash2" in names
    kind = kind_hint
    if kind is None:
        if name.endswith(".work") or name.startswith("work-"):
            kind = "work"
        elif has_quant:
            kind = "exl3"
        elif has_config and any(n.endswith(".safetensors") for n in names):
            kind = "hf"
        else:
            kind = "dir"
    return {"name": name, "path": os.path.relpath(path, REPO), "kind": kind,
            "size": size, "has_draft": has_draft, "has_config": has_config,
            "n_files": len(names)}


async def api_models(_request):
    out = []
    if os.path.isdir(TEST_MODELS):
        for e in sorted(os.scandir(TEST_MODELS), key=lambda e: e.name):
            if e.is_dir() and e.name != "sources":
                m = _model_entry(e.path)
                if m:
                    out.append(m)
    src = os.path.join(TEST_MODELS, "sources")
    if os.path.isdir(src):
        for e in sorted(os.scandir(src), key=lambda e: e.name):
            if e.is_dir():
                m = _model_entry(e.path, kind_hint="source")
                if m:
                    out.append(m)
    return web.json_response({"models": out})


async def api_model_delete(request):
    """Delete a model directory listed in the Models section.

    Strictly confined to test_models/ (the only tree the Models section
    exposes). Symlinks are removed as links (target untouched); real dirs
    are rmtree'd. The frontend gates this behind a confirm() dialog.
    """
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON")
    rel = (body.get("path") or "").strip()
    if not rel:
        return err("path required")
    target = os.path.abspath(os.path.join(REPO, rel))
    if not target.startswith(TEST_MODELS + os.sep):
        return err("path must be inside test_models/", 403)
    if not os.path.lexists(target):
        return err("not found", 404)
    import shutil
    if os.path.islink(target):
        os.remove(target)
    elif os.path.isdir(target):
        shutil.rmtree(target)
    else:
        return err("not a directory", 400)
    return web.json_response({"deleted": rel})


async def api_jobs(_request):
    jobs = JOBS.list()
    legacy = []
    if os.path.exists(LEGACY_JOBS):
        with open(LEGACY_JOBS) as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                legacy.append({
                    "id": None, "kind": "quant", "name": r.get("name", ""),
                    "status": r.get("status", ""), "started": r.get("time", ""),
                    "ended": None, "log": r.get("log"), "meta": {
                        "bits": r.get("bits"), "hq": r.get("hq"),
                        "out": r.get("out"), "model": r.get("model"),
                        "elapsed_s": r.get("elapsed_s"), "legacy": True},
                })
    legacy.reverse()
    return web.json_response({"jobs": jobs, "legacy": legacy})


async def api_job_log(request):
    job_id = request.match_info["id"]
    tail = int(request.query.get("tail", 120_000))
    job = JOBS.get(job_id)
    if not job:
        return err("unknown job", 404)
    text = JOBS.tail_log(job_id, tail)
    return web.json_response({"id": job_id, "status": job["status"],
                              "log": text, "log_path": job.get("log")})


async def api_job_start(request):
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON")
    spec = spec_by_key(body.get("spec", ""))
    if not spec:
        return err("unknown spec")
    try:
        params = clean_params(spec, body.get("params") or {})
        kind, name, argv, meta, env_extra = build_job(spec, params)
    except ValueError as e:
        return err(str(e))
    job = JOBS.start(kind, name, argv, meta=meta, env_extra=env_extra)
    return web.json_response({"job": JOBS.get(job["id"])})


async def api_job_kill(request):
    job, e = JOBS.kill(request.match_info["id"])
    if e and not job:
        return err(e, 404)
    return web.json_response({"job": job, **({"error": e} if e else {})})


async def api_serve_health(request):
    url = request.query.get("url", "http://127.0.0.1:8888").rstrip("/")
    if not url.startswith("http"):
        return err("url must be http(s)")
    try:
        async with ClientSession(timeout=ClientTimeout(total=4)) as cs:
            async with cs.get(url + "/health") as r:
                data = await r.json()
        return web.json_response({"url": url, "up": True, "health": data})
    except Exception as e:
        return web.json_response({"url": url, "up": False, "error": str(e)})


async def api_chat(request):
    """Stream-proxy a chat completion to a running serve_openai endpoint."""
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON")
    base = (body.get("base_url") or "http://127.0.0.1:8888").rstrip("/")
    if not base.startswith("http"):
        return err("base_url must be http(s)")
    payload = body.get("payload") or {}
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream" if payload.get("stream") else "application/json",
        "Cache-Control": "no-cache"})
    try:
        async with ClientSession(timeout=ClientTimeout(total=None, sock_read=300)) as cs:
            async with cs.post(base + "/v1/chat/completions", json=payload) as r:
                if payload.get("stream"):
                    await resp.prepare(request)
                    async for chunk in r.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                data = await r.read()
                return web.Response(body=data, status=r.status,
                                    content_type="application/json")
    except Exception as e:
        if resp.prepared:
            await resp.write(f'data: {json.dumps({"error": {"message": str(e)}})}\n\n'.encode())
            await resp.write_eof()
            return resp
        return err(f"server unreachable: {e}", 502)


async def api_readfile(request):
    """Serve a log file's tail for legacy jobs.jsonl records. Repo paths only."""
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON")
    path = os.path.abspath(body.get("path") or "")
    if not path.startswith(REPO + os.sep):
        return err("path must be inside the repo", 403)
    if not os.path.isfile(path):
        return err("file not found", 404)
    size = os.path.getsize(path)
    with open(path, "rb") as fp:
        if size > 400_000:
            fp.seek(size - 400_000)
        data = fp.read()
    return web.json_response({"path": path, "text": data.decode("utf-8", errors="replace")})


# ------------------------------------------------------------------------ main

async def index(_request):
    return web.FileResponse(os.path.join(STATIC, "index.html"))


def main():
    ap = argparse.ArgumentParser(description="EXL3 web console")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind (0.0.0.0 to reach from the LAN)")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/api/system", api_system)
    app.router.add_get("/api/specs", api_specs)
    app.router.add_get("/api/models", api_models)
    app.router.add_post("/api/models/delete", api_model_delete)
    app.router.add_get("/api/jobs", api_jobs)
    app.router.add_get("/api/jobs/{id}/log", api_job_log)
    app.router.add_post("/api/jobs/start", api_job_start)
    app.router.add_post("/api/jobs/{id}/kill", api_job_kill)
    app.router.add_get("/api/serve/health", api_serve_health)
    app.router.add_post("/api/chat", api_chat)
    app.router.add_post("/api/readfile", api_readfile)
    app.router.add_static("/", STATIC, show_index=False)

    print(f" == EXL3 web console on http://{args.host}:{args.port}")
    print(f" == repo: {REPO}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
