"""Scoring pipeline over a run directory.

The benchmark unit is a manifest ROW = (clip_id, bucket); a clip can appear in both
buckets (e.g. Laughter perception + generation) and then shares one
`runs/<name>/<clip_id>/` directory (same `run.json`, same `agent.wav`). Scoring
therefore iterates manifest rows, not directories, and writes per-row outputs under
a per-bucket subdirectory:

    runs/<name>/<clip_id>/agent.wav                 (shared by both rows)
    runs/<name>/<clip_id>/agent.words.json          (ASR cache, clip level)
    runs/<name>/<clip_id>/<bucket>/timing.json
    runs/<name>/<clip_id>/<bucket>/judge.json
    runs/<name>/<clip_id>/<bucket>/judge/<axis>_<judge>.json

`score_timing(run_dir)`: ASR -> segments -> overlap / yield / timing for every row
whose clip dir exists; writes `timing_summary.json` at the run root (n = rows).
`score(run_dir, judge, axes)` (WS5) runs `score_timing` then the perception LLM
judge over every perception row, writing `judge_summary.json` (n = rows).
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev

from ovfdb.schema import (
    AgentRole,
    AgentRun,
    Axis,
    Bucket,
    DynamicType,
    Sample,
    TimingClass,
    read_manifest,
    run_dir_names,
)
from ovfdb.score import asr, vad
from ovfdb.score import judge as judge_mod
from ovfdb.score.segments import format_partition, merge_vad_and_words, partition, words_to_segments
from ovfdb.score.timing import TimingResult, aggregate, evaluate, overlap_summary, yield_analysis

log = logging.getLogger(__name__)
DEFAULT_MANIFEST = "manifest/ovfdb_v0.jsonl"


@lru_cache(maxsize=8)
def _samples(manifest: str) -> tuple[Sample, ...]:
    return tuple(read_manifest(manifest))


def row_dir(clip_dir: Path, bucket: Bucket | str) -> Path:
    """`runs/<name>/<clip_id>/<bucket>/` — per-row output directory."""
    return Path(clip_dir) / Bucket(bucket).value


def row_key(sample: Sample) -> str:
    return f"{sample.clip_id}/{sample.bucket.value}"


def _load_run(path: Path) -> AgentRun:
    d = json.loads(path.read_text())
    known = AgentRun.__dataclass_fields__
    return AgentRun(**{k: v for k, v in d.items() if k in known})


def _resolve(p: str | None, base: Path, default: str) -> Path:
    if not p:
        return base / default
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    if (base / q.name).exists():
        return base / q.name
    return q


def agent_segments(wav: Path, words: list, clip_id: str = "", *, vad_backend: str = "auto") -> tuple[list, dict]:
    """Agent speech segments = VAD regions on `wav` with ASR `words` attached (docs/protocol.md §4).

    Falls back to `words_to_segments(words)` when the VAD cannot run (unreadable / empty wav,
    missing backend). Returns (segments, info) where info records which path was used.
    """
    try:
        det = vad.detect(wav, backend=vad_backend)
    except Exception as e:  # noqa: BLE001 - any VAD failure degrades to the ASR-only path
        log.warning("%s: VAD failed (%s: %s); segments from ASR words only", clip_id or wav, type(e).__name__, e)
        return words_to_segments(words), {"source": "asr", "vad_backend": None, "n_regions": None,
                                          "error": f"{type(e).__name__}: {e}"}
    info = {"source": "vad+asr", "vad_backend": det.backend, "n_regions": len(det.regions)}
    if det.error:
        info["error"] = det.error
    return merge_vad_and_words(det.regions, words), info


def score_clip(clip_dir: Path, sample: Sample, run: AgentRun, *, asr_model: str = asr.DEFAULT_MODEL,
               asr_backend: str = asr.DEFAULT_BACKEND, vad_backend: str = "auto") -> dict:
    """ASR + VAD + derived signals + timing for one clip. Returns the timing.json payload.

    `vad_backend` reaches `agent_segments`, which already accepts it. Worth exposing because
    the right detector depends on what produced the agent track: "auto" prefers silero, which
    is built for noisy human audio, and on synthesised speech against digital silence it can
    miss a clearly audible utterance entirely. Measured on one clip, a 420 ms backchannel at
    RMS ~2000 against a zero floor: silero found no region, energy found 17.88-18.30. With no
    region to anchor it the word falls through to its ASR timestamps, which whisper had given
    a 1.40 s span starting 1.09 s before the sound, and `is_backchannel` then rejects it on
    duration and counts it as taking the floor -- so a backchannel scores as its opposite.
    """
    wav = _resolve(run.agent_audio_path, clip_dir, "agent.wav")
    words = asr.load_or_transcribe(wav, model=asr_model, backend=asr_backend)
    segs, seg_info = agent_segments(wav, words, sample.clip_id, vad_backend=vad_backend)
    ds, de, end = sample.dynamic_start_s, sample.dynamic_end_s, sample.duration_s
    ov = overlap_summary(segs, ds, de, end)
    parts = partition(segs, ds, de, end)
    out: dict = {
        "clip_id": sample.clip_id,
        "provider": run.provider,
        "mode": run.mode,
        "bucket": sample.bucket.value,
        "dynamic_type": sample.dynamic_type.value,
        "dynamic_start_s": ds,
        "dynamic_end_s": de,
        "duration_s": end,
        "agent_role": sample.agent_role.value,
        "asr": {"model": asr_model, "backend": asr_backend, "n_words": len(words)},
        "segments_source": seg_info,
        "segments": [asdict(s) for s in segs],
        "transcript": " ".join(s.text for s in segs if s.text) or "[no speech]",
        "overlap_line": ov.line(),
        "overlap": {
            "total_in_window_s": round(ov.total_in_window_s, 3),
            "pre_activity": round(ov.pre_activity, 4),
            "in_activity": round(ov.in_activity, 4),
            "post_activity": round(ov.post_activity, 4),
        },
        "partition_block": format_partition(parts),
        "yield_block": None,
        "timing_class": None,
        "timing": None,
    }
    if sample.agent_role is AgentRole.YIELDING:
        ya = yield_analysis(segs, ds, de, end)
        out["yield_block"] = ya.block()
        out["yield"] = asdict(ya)
    cls = sample.timing_class
    if cls is not None:
        r = evaluate(segs, ds, de, cls, agent_speaks_first=sample.agent_speaks_first, clip_end=end)
        out["timing_class"] = cls.value
        out["timing"] = {
            "passed": r.passed,
            "takeover": r.takeover,
            "expected_takeover": r.expected_takeover,
            "latency_ms": r.latency_ms,
            "details": r.details,
        }
    return out


def _result_from_payload(p: dict) -> TimingResult:
    t = p["timing"]
    return TimingResult(timing_class=TimingClass(p["timing_class"]), passed=t["passed"], takeover=t["takeover"],
                        expected_takeover=t["expected_takeover"], latency_ms=t["latency_ms"],
                        details=t.get("details", {}))


def iter_clip_dirs(run_dir: Path):
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and (d / "run.json").exists():
            yield d


def iter_rows(run_dir: Path, manifest: str) -> tuple[list[tuple[Sample, Path, AgentRun]], list[dict]]:
    """Manifest rows with an existing clip dir, in manifest order, plus `skipped` notes.

    Returns ([(sample, clip_dir, run), ...], skipped). Rows whose `runs/<name>/<clip_id>/run.json`
    is missing are not listed (the run never produced them); run dirs absent from the manifest are
    reported in `skipped`.
    """
    rows: list[tuple[Sample, Path, AgentRun]] = []
    seen: set[str] = set()
    runs: dict[str, AgentRun] = {}
    samples = _samples(manifest)
    dir_names = run_dir_names(samples)      # <clip_id>, or <clip_id>__<bucket> when buckets swap the user
    for sample in samples:
        name = dir_names[(sample.clip_id, sample.bucket.value)]
        clip_dir = run_dir / name
        if not (clip_dir / "run.json").exists():
            continue
        if name not in runs:
            runs[name] = _load_run(clip_dir / "run.json")
        rows.append((sample, clip_dir, runs[name]))
        seen.add(name)
    skipped = [{"clip_id": d.name, "bucket": None, "reason": f"clip not in manifest {manifest}"}
               for d in iter_clip_dirs(run_dir) if d.name not in seen]
    return rows, skipped


def score_timing(run_dir: str | Path, manifest: str | None = None, *, asr_model: str = asr.DEFAULT_MODEL,
                 asr_backend: str = asr.DEFAULT_BACKEND, force: bool = False) -> dict:
    """ASR + timing metrics for every manifest row in `run_dir`; writes `<clip>/<bucket>/timing.json`
    and `timing_summary.json` (n = rows, keyed by (clip_id, bucket)).

    The manifest is `manifest` if given, else the path stored in `run.json` meta (`meta["manifest"]`),
    else `<run_dir>/manifest.jsonl`, else `manifest/ovfdb_v0.jsonl`. Rows without a timing class
    still get a timing.json (segments, overlap line, partition) with `timing: null`. ASR runs once
    per clip (cached in `agent.words.json`) even when the clip has rows in both buckets.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    man = _resolve_manifest(run_dir, manifest)
    if not man or not Path(man).exists():
        raise FileNotFoundError(f"manifest not found: {man!r}")
    rows, skipped = iter_rows(run_dir, man)
    per_row: list[dict] = []
    by_dynamic: dict[str, list[TimingResult]] = {}
    for sample, clip_dir, run in rows:
        out_dir = row_dir(clip_dir, sample.bucket)
        out_path = out_dir / "timing.json"
        if out_path.exists() and not force:
            payload = json.loads(out_path.read_text())
        else:
            payload = score_clip(clip_dir, sample, run, asr_model=asr_model, asr_backend=asr_backend)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        per_row.append(payload)
        if payload.get("timing"):
            by_dynamic.setdefault(payload["dynamic_type"], []).append(_result_from_payload(payload))
    summary = {
        "run_dir": str(run_dir),
        "manifest": man,
        "n_rows": len(per_row),
        "n_clips": len({p["clip_id"] for p in per_row}),
        "n_timed": sum(1 for p in per_row if p.get("timing")),
        "skipped": skipped,
        "aggregate": aggregate(by_dynamic),
        "rows": [
            {"clip_id": p["clip_id"], "bucket": p["bucket"], "dynamic_type": p["dynamic_type"],
             "timing_class": p["timing_class"],
             "passed": p["timing"]["passed"] if p.get("timing") else None,
             "latency_ms": p["timing"]["latency_ms"] if p.get("timing") else None}
            for p in per_row
        ],
    }
    (run_dir / "timing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _resolve_manifest(run_dir: Path, manifest: str | None) -> str | None:
    """Explicit arg -> run.json meta -> <run_dir>/manifest.jsonl -> manifest/ovfdb_v0.jsonl."""
    if manifest and Path(manifest).exists():
        return str(manifest)
    for clip_dir in iter_clip_dirs(run_dir):
        m = _load_run(clip_dir / "run.json").meta.get("manifest")
        if m and Path(m).exists():
            return str(m)
        break
    for cand in (run_dir / "manifest.jsonl", Path(DEFAULT_MANIFEST)):
        if cand.exists():
            return str(cand)
    return manifest


def _parse_axes(axes: str | None) -> set[Axis] | None:
    if axes in (None, "", "all"):
        return None
    return {Axis(a.strip()) for a in str(axes).split(",") if a.strip()}


def _stats(values: list[float]) -> dict:
    n = len(values)
    return {"mean": mean(values) if n else None, "sd": stdev(values) if n > 1 else (0.0 if n else None), "n": n}


def _clip_judge_payload(sample: Sample, run: AgentRun, judge: str, results: dict[str, judge_mod.JudgeResult],
                        skipped_axes: list[str], existing: dict | None) -> dict:
    axes = {a: {"rating": r.rating, "analysis": r.analysis, "reasoning": r.reasoning, "rubric": r.rubric,
                "cached": r.cached, "usage": r.usage} for a, r in results.items()}
    overall = mean(r.rating for r in results.values()) if results else None
    block = {"judge": judge, "payload_version": judge_mod.PAYLOAD_VERSION, "axes": axes, "overall": overall,
             "skipped_axes": skipped_axes}
    by_judge = dict((existing or {}).get("by_judge") or {})
    by_judge[judge] = block
    return {
        "clip_id": sample.clip_id, "provider": run.provider, "mode": run.mode, "bucket": sample.bucket.value,
        "dynamic_type": sample.dynamic_type.value, "agent_role": sample.agent_role.value,
        "has_caption": bool((sample.av_caption or "").strip()),
        **block, "by_judge": by_judge,
    }


def score(run_dir: str | Path, judge: str = judge_mod.DEFAULT_JUDGE, axes: str = "all",
          manifest: str | None = None, *, force: bool = False) -> dict:
    """WS5: `score_timing` (reuses `<bucket>/timing.json`) + perception judge on every perception row.

    `axes` is "all" or a comma list of axis values (e.g. "fluency,conversational_flow"); generation
    axes and generation-bucket rows are skipped with a note. Writes `<clip>/<bucket>/judge.json`
    (results of this judge at top level, every judge so far under `by_judge`) and
    `<run_dir>/judge_summary.json`: per-axis mean/sd/n per dynamic and overall (n = rows), per-row
    `overall` = mean of its axes, token usage.
    """
    run_dir = Path(run_dir)
    manifest = _resolve_manifest(run_dir, manifest)
    timing_summary = score_timing(run_dir, manifest=manifest, force=force)   # ASR cache is kept either way
    manifest = timing_summary["manifest"]
    wanted = _parse_axes(axes)
    rows_out: list[dict] = []
    skipped: list[dict] = list(timing_summary.get("skipped", []))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "n_cached": 0}
    rows, _ = iter_rows(run_dir, manifest)
    for sample, clip_dir, run in rows:
        out_dir = row_dir(clip_dir, sample.bucket)
        tpath = out_dir / "timing.json"
        if not tpath.exists():
            continue                      # score_timing already recorded why it was skipped
        if sample.bucket is not Bucket.PERCEPTION:
            skipped.append({"clip_id": sample.clip_id, "bucket": sample.bucket.value,
                            "reason": f"{sample.bucket.value} bucket: perception judge n/a"})
            continue
        timing_json = json.loads(tpath.read_text())
        sel = [a for a in sample.axes if wanted is None or a in wanted]
        skipped_axes = [a.value for a in sample.axes if a not in judge_mod.PERCEPTION_AXES]
        results = judge_mod.score_sample(sample, timing_json, judge, axes=sel, cache_dir=out_dir / "judge",
                                         force=force)
        for r in results.values():
            if r.cached:
                usage["n_cached"] += 1
            else:
                for k in ("prompt_tokens", "completion_tokens", "n_calls"):
                    usage[k] += int(r.usage.get(k, 0) or 0)
        jpath = out_dir / "judge.json"
        existing = json.loads(jpath.read_text()) if jpath.exists() else None
        payload = _clip_judge_payload(sample, run, judge, results, skipped_axes, existing)
        jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        rows_out.append(payload)

    axis_names = sorted({a for c in rows_out for a in c["axes"]})
    per_dynamic: dict[str, dict] = {}
    for dyn in sorted({c["dynamic_type"] for c in rows_out}):
        cs = [c for c in rows_out if c["dynamic_type"] == dyn]
        per_dynamic[dyn] = {a: _stats([c["axes"][a]["rating"] for c in cs if a in c["axes"]]) for a in axis_names
                            if any(a in c["axes"] for c in cs)}
        per_dynamic[dyn]["overall"] = _stats([c["overall"] for c in cs if c["overall"] is not None])
    summary = {
        "run_dir": str(run_dir),
        "judge": judge,
        "payload_version": judge_mod.PAYLOAD_VERSION,
        "manifest": manifest,
        "n_rows": timing_summary.get("n_rows", 0),
        "n_clips": timing_summary.get("n_clips", 0),
        "n_judged": len(rows_out),
        "skipped": skipped,
        "per_axis": {a: _stats([c["axes"][a]["rating"] for c in rows_out if a in c["axes"]]) for a in axis_names},
        "overall": _stats([c["overall"] for c in rows_out if c["overall"] is not None]),
        "per_dynamic": per_dynamic,
        "usage": {**usage, "est_cost_usd": judge_mod.estimate_cost_usd(usage, judge)},
        "timing": timing_summary.get("aggregate", {}),
        "rows": [{"clip_id": c["clip_id"], "bucket": c["bucket"], "dynamic_type": c["dynamic_type"],
                  "overall": c["overall"], "axes": {a: v["rating"] for a, v in c["axes"].items()}}
                 for c in rows_out],
    }
    (run_dir / "judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("judged %d/%d rows with %s; overall %s", len(rows_out), summary["n_rows"], judge, summary["overall"])
    return summary


# --- generation (WS6c) -----------------------------------------------------------------------

def _gen_judge_payload(sample: Sample, run: AgentRun, judge: str, results: dict, frames: dict,
                       existing: dict | None) -> dict:
    """judge.json for a generation row: this judge's generation axes at top level (merged into any axes the
    same judge already wrote for the row), every judge so far under `by_judge`."""
    from ovfdb.score import judge_generation as gen_mod

    axes = {a: {"rating": r.rating, "analysis": r.analysis, "reasoning": r.reasoning, "rubric": r.rubric,
                "cached": r.cached, "usage": r.usage, "cue_produced": r.cue_produced, "cue_timing": r.cue_timing,
                "consistency_fixes": r.consistency_fixes, "n_frames": r.n_frames} for a, r in results.items()}
    prev = {}
    if existing and existing.get("judge") == judge:
        prev = dict(existing.get("axes") or {})
    elif existing:
        prev = dict(((existing.get("by_judge") or {}).get(judge) or {}).get("axes") or {})
    prev.update(axes)
    axes = prev
    overall = mean(v["rating"] for v in axes.values()) if axes else None
    block = {"judge": judge, "payload_version": gen_mod.PAYLOAD_VERSION, "axes": axes, "overall": overall,
             "skipped_axes": [a.value for a in sample.axes if a.value not in axes],
             "frames": {k: frames.get(k) for k in ("n", "fps", "width", "source", "agent_video", "composite")}}
    by_judge = dict((existing or {}).get("by_judge") or {})
    by_judge[judge] = block
    return {
        "clip_id": sample.clip_id, "provider": run.provider, "mode": run.mode, "bucket": sample.bucket.value,
        "dynamic_type": sample.dynamic_type.value, "agent_role": sample.agent_role.value,
        "has_caption": bool((sample.av_caption or "").strip()),
        **block, "by_judge": by_judge,
    }


def _has_generation_axes(existing: dict | None, judge: str, wanted: list[Axis]) -> bool:
    if not existing:
        return False
    block = existing if existing.get("judge") == judge else ((existing.get("by_judge") or {}).get(judge) or {})
    axes = block.get("axes") or {}
    return bool(wanted) and all(a.value in axes for a in wanted)


def score_generation(run_dir: str | Path, judge: str = judge_mod.DEFAULT_JUDGE, axes: str = "all",
                     manifest: str | None = None, *, force: bool = False, fps: float | None = None,
                     max_frames: int | None = None, width: int | None = None, limit: int = 0,
                     dry_run: bool = False, rebuild_frames: bool = False, dynamics: str | None = None) -> dict:
    """WS6c: generation (video-out) judge over every generation row of `run_dir` that has an agent video.

    Reuses `<bucket>/timing.json` (`score_timing` fills any that are missing). Per row: side-by-side
    composite + sampled frames cached under `<row>/gen/` (`fps`, `max_frames`, `width` default to the
    paper's 8 / 600 / 256), one vision-judge call per axis (cache `<row>/judge/<axis>_<judge>.json`),
    `judge.json` written next to timing.json. Rows whose judge.json already has every wanted generation
    axis from this judge are skipped unless `force` (which also re-judges cached axes; `rebuild_frames`
    re-encodes the media). `dry_run` builds the media and prints the text payload without calling the
    judge. `dynamics` (comma list of dynamic types) restricts the rows, e.g. to the dynamics whose human
    reference exists. Writes `<run_dir>/judge_summary_generation.json` (the perception summary is left alone).
    """
    from ovfdb.score import judge_generation as gen_mod

    run_dir = Path(run_dir)
    manifest = _resolve_manifest(run_dir, manifest)
    timing_summary = score_timing(run_dir, manifest=manifest)
    manifest = timing_summary["manifest"]
    wanted = _parse_axes(axes)
    fps = gen_mod.FPS if fps is None else fps
    max_frames = gen_mod.MAX_FRAMES if max_frames is None else max_frames
    width = gen_mod.FRAME_WIDTH if width is None else width
    rows_out: list[dict] = []
    skipped: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "n_calls": 0, "n_cached": 0}
    rows, _ = iter_rows(run_dir, manifest)
    only = {DynamicType(d.strip()) for d in str(dynamics).split(",") if d.strip()} if dynamics else None
    rows = [r for r in rows if r[0].bucket is Bucket.GENERATION and (only is None or r[0].dynamic_type in only)]
    n_done = 0
    for sample, clip_dir, run in rows:
        out_dir = row_dir(clip_dir, sample.bucket)
        tpath = out_dir / "timing.json"
        if not tpath.exists():
            skipped.append({"clip_id": sample.clip_id, "bucket": sample.bucket.value, "reason": "no timing.json"})
            continue
        sel = [a for a in sample.axes if a in gen_mod.GENERATION_AXES and (wanted is None or a in wanted)]
        jpath = out_dir / "judge.json"
        existing = json.loads(jpath.read_text()) if jpath.exists() else None
        if not force and not dry_run and _has_generation_axes(existing, judge, sel):
            payload = existing if existing.get("judge") == judge else None
            if payload is None:                      # another judge is at top level: read ours from by_judge
                block = (existing.get("by_judge") or {})[judge]
                payload = {**existing, **block}
            payload["_reused"] = True
            rows_out.append(payload)
            usage["n_cached"] += len(sel)
            continue
        if limit and n_done >= limit:
            break
        try:
            frames = gen_mod.prepare_frames(sample, run, clip_dir, out_dir / "gen", fps=fps, max_frames=max_frames,
                                            width=width, force=rebuild_frames)
        except gen_mod.GenerationJudgeError as e:
            log.warning("%s/generation: skipped (%s)", sample.clip_id, e)
            skipped.append({"clip_id": sample.clip_id, "bucket": sample.bucket.value, "reason": str(e)})
            continue
        except subprocess.CalledProcessError as e:
            msg = (e.stderr or "").strip().splitlines()[-1:] or [str(e)]
            log.warning("%s/generation: ffmpeg failed (%s)", sample.clip_id, msg[0])
            skipped.append({"clip_id": sample.clip_id, "bucket": sample.bucket.value, "reason": f"ffmpeg: {msg[0]}"})
            continue
        timing_json = json.loads(tpath.read_text())
        n_done += 1
        if dry_run:
            text = gen_mod.build_user_text(sample, timing_json, frames)
            print(f"--- {sample.clip_id}/generation ({sample.dynamic_type.value}): {frames['n']} frames @ "
                  f"{frames['fps']:.2f} fps, {frames['width']} px wide, agent video = {frames['source']}\n{text}\n")
            rows_out.append({"clip_id": sample.clip_id, "bucket": "generation", "dynamic_type": sample.dynamic_type.value,
                             "axes": {}, "overall": None, "frames": frames, "dry_run": True})
            continue
        results = gen_mod.score_sample(sample, timing_json, frames, judge, axes=sel, cache_dir=out_dir / "judge",
                                       force=force)
        for r in results.values():
            if r.cached:
                usage["n_cached"] += 1
            else:
                for k in ("prompt_tokens", "completion_tokens", "n_calls"):
                    usage[k] += int(r.usage.get(k, 0) or 0)
        payload = _gen_judge_payload(sample, run, judge, results, frames, existing)
        jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        rows_out.append(payload)
        row_usage = {k: sum(int(r.usage.get(k, 0) or 0) for r in results.values()) for k in ("prompt_tokens", "completion_tokens")}
        log.info("%s/generation %s: %s (frames %d, tokens in %d / out %d, ~$%.3f)", sample.clip_id,
                 sample.dynamic_type.value, {a: r.rating for a, r in results.items()}, frames["n"],
                 row_usage["prompt_tokens"], row_usage["completion_tokens"],
                 gen_mod.estimate_cost_usd(row_usage, judge) or 0.0)

    judged = [c for c in rows_out if c.get("axes")]
    axis_names = sorted({a for c in judged for a in c["axes"]})
    per_dynamic: dict[str, dict] = {}
    for dyn in sorted({c["dynamic_type"] for c in judged}):
        cs = [c for c in judged if c["dynamic_type"] == dyn]
        per_dynamic[dyn] = {a: _stats([c["axes"][a]["rating"] for c in cs if a in c["axes"]]) for a in axis_names
                            if any(a in c["axes"] for c in cs)}
        per_dynamic[dyn]["overall"] = _stats([c["overall"] for c in cs if c["overall"] is not None])
    cue = [c["axes"][Axis.CUE_APPROPRIATENESS.value] for c in judged if Axis.CUE_APPROPRIATENESS.value in c["axes"]]
    cost = gen_mod.estimate_cost_usd(usage, judge)
    n_fresh = sum(1 for c in judged if not c.get("_reused") and any(not v.get("cached") for v in c["axes"].values()))
    summary = {
        "run_dir": str(run_dir),
        "judge": judge,
        "payload_version": gen_mod.PAYLOAD_VERSION,
        "manifest": manifest,
        "frames": {"fps": fps, "max_frames": max_frames, "width": width},
        "dynamics": sorted(d.value for d in only) if only else None,
        "n_generation_rows": len(rows),
        "n_judged": len(judged),
        "skipped": skipped,
        "per_axis": {a: _stats([c["axes"][a]["rating"] for c in judged if a in c["axes"]]) for a in axis_names},
        "overall": _stats([c["overall"] for c in judged if c["overall"] is not None]),
        "cue": {
            "n": len(cue),
            "produced_rate": (sum(1 for v in cue if v.get("cue_produced")) / len(cue)) if cue else None,
            "timing": {t: sum(1 for v in cue if v.get("cue_timing") == t) for t in gen_mod.TIMING_VALUES},
            "n_consistency_fixes": sum(1 for v in cue if v.get("consistency_fixes")),
        },
        "per_dynamic": per_dynamic,
        "usage": {**usage, "est_cost_usd": cost, "n_fresh_rows": n_fresh,
                  "est_cost_per_fresh_row_usd": (cost / n_fresh) if (cost is not None and n_fresh) else None},
        "timing": timing_summary.get("aggregate", {}),
        "rows": [{"clip_id": c["clip_id"], "bucket": c["bucket"], "dynamic_type": c["dynamic_type"],
                  "overall": c["overall"], "axes": {a: v["rating"] for a, v in c["axes"].items()},
                  "cue_produced": (c["axes"].get(Axis.CUE_APPROPRIATENESS.value) or {}).get("cue_produced"),
                  "cue_timing": (c["axes"].get(Axis.CUE_APPROPRIATENESS.value) or {}).get("cue_timing"),
                  "n_frames": next((v.get("n_frames") for v in c["axes"].values() if v.get("n_frames")),
                                   (c.get("frames") or {}).get("n")),
                  "usage": {k: sum(int((v.get("usage") or {}).get(k, 0) or 0) for v in c["axes"].values())
                            for k in ("prompt_tokens", "completion_tokens")}}
                 for c in rows_out],
    }
    if not dry_run:
        (run_dir / "judge_summary_generation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("generation: judged %d/%d rows with %s; overall %s; skipped %d", len(judged),
             summary["n_generation_rows"], judge, summary["overall"], len(skipped))
    return summary
