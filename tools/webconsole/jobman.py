"""
Job manager for the EXL3 web console.

Runs CLI scripts (quantize / eval / tools / serve) as detached subprocesses
with logs under runs/webui/logs/, and keeps an append-only registry in
runs/webui/jobs.jsonl. Detached children (new session) survive a console
restart; on restart their status is re-derived from the pid.

Design notes:
  * append-only JSONL registry: one record per state change, last record per
    id wins (crash-safe, no rewrite races with concurrent readers)
  * the console never runs anything heavy itself — every GPU job is a child
    process of the repo's own CLI scripts
"""
import json, os, signal, subprocess, time, uuid

STATUSES = ("running", "done", "failed", "killed", "exited")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class JobManager:
    def __init__(self, repo_root):
        self.repo = repo_root
        self.base = os.path.join(repo_root, "runs", "webui")
        self.logs = os.path.join(self.base, "logs")
        self.registry = os.path.join(self.base, "jobs.jsonl")
        os.makedirs(self.logs, exist_ok=True)
        self.procs = {}  # job_id -> subprocess.Popen (only while this server owns it)

    # ---- registry ----

    def _append(self, rec):
        with open(self.registry, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _records(self):
        jobs = {}
        if os.path.exists(self.registry):
            with open(self.registry) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    jid = r.get("id")
                    if not jid:
                        continue
                    if jid not in jobs:
                        jobs[jid] = r
                    else:
                        jobs[jid].update(r)
        return jobs

    @staticmethod
    def _alive(pid):
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, OSError):
            return False

    def _refresh(self, job):
        """Recompute live status from the OS."""
        if job.get("status") != "running":
            return job
        proc = self.procs.get(job["id"])
        if proc is not None:
            # we own the Popen: poll() reaps it (kill(pid,0) would still see
            # the zombie and wrongly report it alive)
            rc = proc.poll()
            if rc is None:
                return job
            self.procs.pop(job["id"], None)
            job["status"] = "done" if rc == 0 else ("killed" if rc in (-signal.SIGTERM, -signal.SIGKILL) else "failed")
            job["exit_code"] = rc
        else:
            # started by a previous server instance: only the pid is known
            if self._alive(job.get("pid")):
                return job
            job["status"] = "exited"
        job["ended"] = job.get("ended") or _now()
        self._append({"id": job["id"], "status": job["status"],
                      "exit_code": job.get("exit_code"), "ended": job["ended"]})
        return job

    def list(self):
        jobs = [self._refresh(j) for j in self._records().values()]
        jobs.sort(key=lambda j: j.get("started", ""), reverse=True)
        return jobs

    def get(self, job_id):
        job = self._records().get(job_id)
        return self._refresh(job) if job else None

    # ---- lifecycle ----

    def start(self, kind, name, argv, meta=None, env_extra=None, cwd=None):
        """argv may be a list, or a callable(log_path) -> (argv, extra_logs)
        so the caller can route child-process logs next to the job log."""
        job_id = uuid.uuid4().hex[:10]
        log_path = os.path.join(self.logs, f"{job_id}.log")
        extra_logs = []
        if callable(argv):
            argv, extra_logs = argv(log_path)
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        lf = open(log_path, "w")
        lf.write(f"$ {' '.join(argv)}\n$ cwd: {cwd or self.repo}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            argv, stdout=lf, stderr=subprocess.STDOUT,
            cwd=cwd or self.repo, env=env,
            start_new_session=True)  # detached: survives console restart, killable by pgid
        job = {
            "id": job_id, "kind": kind, "name": name, "argv": argv,
            "pid": proc.pid, "log": log_path, "status": "running",
            "started": _now(), "ended": None, "exit_code": None,
            "meta": meta or {}, "extra_logs": extra_logs,
        }
        self.procs[job_id] = proc
        self._append(job)
        return job

    def kill(self, job_id):
        job = self.get(job_id)
        if not job or job.get("status") != "running":
            return job, "job is not running"
        pid = job.get("pid")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as e:
            return job, f"SIGTERM failed: {e}"
        deadline = time.time() + 6
        while time.time() < deadline and self._alive(pid):
            time.sleep(0.2)
        if self._alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        job["status"] = "killed"
        job["ended"] = _now()
        self._append({"id": job_id, "status": "killed", "ended": job["ended"]})
        self.procs.pop(job_id, None)
        return job, None

    # ---- logs ----

    def tail_log(self, job_id, limit=200_000):
        """Combined tail of the job log and any extra_logs (e.g. convert.py's
        own log file), capped at ~limit chars total."""
        job = self._records().get(job_id)
        if not job:
            return None
        files = [job.get("log")] + list(job.get("extra_logs") or [])
        parts, budget = [], limit
        for path in reversed(files):
            if budget <= 0 or not path or not os.path.exists(path):
                continue
            size = os.path.getsize(path)
            with open(path, "rb") as fp:
                if size > budget:
                    fp.seek(size - budget)
                parts.append(fp.read().decode("utf-8", errors="replace"))
            budget -= len(parts[-1])
        parts.reverse()
        return "".join(parts)
