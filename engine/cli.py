"""Command-line runner for the reconstruction engine.

    python cli.py analyze <video|url> [--preset maximum] [--out DIR]
    python cli.py demo   [--out DIR]     # generate + analyze a synthetic clip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from synthesia import analyze, Options
from synthesia.exporters import write_midi, write_csv, write_json, write_musicxml, notes_to_json
from synthesia import videoio


def _progress(d):
    bar = int(d.get("progress", 0) * 30)
    sys.stderr.write(f"\r[{'#'*bar:<30}] {d['stage']:<11} {d['message'][:60]:<60}")
    sys.stderr.flush()
    if d.get("stage") == "done":
        sys.stderr.write("\n")


def _export(result, out_dir, truth=None):
    os.makedirs(out_dir, exist_ok=True)
    write_midi(os.path.join(out_dir, "reconstruction.mid"), result.notes, result.tempo)
    write_csv(os.path.join(out_dir, "notes.csv"), result.notes)
    write_musicxml(os.path.join(out_dir, "score.musicxml"), result.notes, result.tempo)
    write_json(os.path.join(out_dir, "project.json"), result.to_dict())
    print(f"\nExports written to {out_dir}")
    r = result.report
    print(f"Notes: {r.notes_total}  high={r.high_confidence} med={r.medium_confidence} "
          f"low={r.low_confidence}  review={r.needs_review}")
    print(f"Onset uncertainty ~±{r.mean_onset_uncertainty_ms:.2f} ms | {r.notes}")
    if result.ground_truth:
        g = result.ground_truth
        print(f"Ground truth: P={g['precision']} R={g['recall']} F1={g['f1']} "
              f"onset_err={g['mean_onset_error_ms']}ms dur_err={g['mean_duration_error_ms']}ms")


def cmd_analyze(args):
    video = args.video
    if video.startswith(("http://", "https://")):
        tmp = tempfile.mkdtemp()
        sys.stderr.write(f"Downloading {video} ...\n")
        video, title = videoio.download(video, tmp)
        sys.stderr.write(f"Downloaded: {title or video}\n")
    opts = Options.for_preset(args.preset)
    opts.detector = args.detector
    result = analyze(video, opts, _progress)
    _export(result, args.out)


def cmd_demo(args):
    from synthesia.synth import generate_video, demo_piece, SynthConfig
    os.makedirs(args.out, exist_ok=True)
    video = os.path.join(args.out, "demo.mp4")
    cfg = SynthConfig()
    truth = generate_video(video, demo_piece(), cfg)
    print(f"Generated {video} ({len(truth)} ground-truth notes)")
    opts = Options.for_preset(args.preset)
    result = analyze(video, opts, _progress, truth=truth)
    _export(result, args.out, truth)


def main():
    p = argparse.ArgumentParser(description="Synthesia reconstruction engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("video")
    a.add_argument("--preset", default="balanced", choices=["fast", "balanced", "maximum"])
    a.add_argument("--detector", default="auto", choices=["auto", "bars", "keylight"])
    a.add_argument("--out", default="out")
    a.set_defaults(func=cmd_analyze)
    d = sub.add_parser("demo")
    d.add_argument("--preset", default="maximum", choices=["fast", "balanced", "maximum"])
    d.add_argument("--out", default="out")
    d.set_defaults(func=cmd_demo)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
