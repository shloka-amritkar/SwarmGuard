"""SwarmGuard command-line interface.

Usage:
    python cli.py list-scenarios
    python cli.py run --scenario single_hijack [--seed 3] [--plot] [--csv out.csv]
    python cli.py detect --scenario rigid_swarm_shift [--seed 5000] [--ml]
    python cli.py train-ml [--runs 300]
    python cli.py calibrate [--seeds 200] [--scenario clean_baseline] [--target 0.01]
    python cli.py sweep --param magnitude --scenario single_hijack --seeds 50 --workers 4
    python cli.py evasion --scenario slow_drift [--seeds 40] [--max-delay 10] [--workers 4]
    python cli.py propagation --scenario clean_baseline [--magnitude 6]
    python cli.py investigate --scenario rigid_swarm_shift [--seed 5000] [--force]
    python cli.py verify data/packages/SG-20261105-001.zip
    python cli.py tamper-test data/packages/SG-20261105-001 --file detections.csv [--byte 100]

All numbers printed are simulation results, not measurements of real drones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# WHY "Agg": draws straight to a PNG file without opening a window, so the
# command also works on machines without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.calibration import calibrate, load_calibration, load_or_default, save_calibration  # noqa: E402
from analysis.evasion_search import stealth_envelope  # noqa: E402
from analysis.experiments import PipelineResult, builtin_values, run_pipeline, sweep  # noqa: E402
from analysis.experiments import summarise as summarise_sweep  # noqa: E402
from analysis.metrics import RunMetrics, compute_metrics  # noqa: E402
from analysis.propagation import propagation_study  # noqa: E402
from attacks.scenarios import Scenario, get_scenario, list_scenarios  # noqa: E402
from config import DetectorConfig  # noqa: E402
from core.schema import SimResult  # noqa: E402
from core.utils import drone_label  # noqa: E402
from database.storage import DEFAULT_DB_PATH, Storage  # noqa: E402
from detection.fusion import describe_pattern, pattern_key  # noqa: E402
from forensics.evidence_package import VerificationReport, package_incident, tamper_test, verify_package  # noqa: E402
from forensics.incident import open_incident, should_open_incident  # noqa: E402
from simulation.swarm import simulate  # noqa: E402

logger = logging.getLogger(__name__)

PLOT_DIR = Path("data") / "plots"


def summarise(result: SimResult) -> dict[str, str]:
    """Key facts about a run, as display strings.

    max offset         = max over steps and drones of |attack_offset|
    mean slot distance = mean over steps and drones of |true_pos - slot_pos|
    """
    offsets = np.linalg.norm(result.stack("attack_offset"), axis=-1)
    slot_err = np.linalg.norm(result.stack("true_pos") - result.stack("slot_pos"), axis=-1)
    attacked = sorted(result.attacked_drones())
    onset = result.attack_onset()
    return {
        "steps": str(result.n_steps),
        "drones": str(result.n_drones),
        "attacked drones": ", ".join(drone_label(i) for i in attacked) or "none",
        "attack onset": f"{onset:.1f} s" if onset is not None else "none",
        "max GPS offset": f"{offsets.max():.2f} m",
        "mean true distance from slots": f"{slot_err.mean():.2f} m",
    }


def plot_result(result: SimResult, scenario: Scenario, path: Path) -> Path:
    """Save a top-down plot: true paths solid, GPS paths dotted, attacked in red."""
    true_pos = result.stack("true_pos")
    gps = result.stack("gps_pos")
    attacked = result.attacked_drones()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    for i in range(result.n_drones):
        colour = "#d62728" if i in attacked else "#4c72b0"
        ax.plot(true_pos[:, i, 0], true_pos[:, i, 1], "-", color=colour, lw=1.2, alpha=0.9)
        ax.plot(gps[:, i, 0], gps[:, i, 1], ":", color=colour, lw=0.8, alpha=0.6)
    ax.plot([], [], "-", color="#4c72b0", label="true path")
    ax.plot([], [], ":", color="#4c72b0", label="reported GPS")
    ax.plot([], [], "-", color="#d62728", label="attacked drone")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(f"{scenario.title} (seed {result.config.seed}) - simulation result")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(result: SimResult, path: Path) -> Path:
    """Write observations and truth side by side, one row per (step, drone)."""
    observed = result.to_observed_dataframe()
    truth = result.to_truth_dataframe()
    table = observed.merge(truth, on=["step", "t", "drone"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def cmd_list_scenarios(_args: argparse.Namespace) -> int:
    """Print every scenario key, title and expected detector outcome."""
    for scenario in list_scenarios():
        expected = ", ".join(f"{k}={v}" for k, v in scenario.expected.items()) or "no attack"
        print(f"{scenario.key:<22} {scenario.title:<34} [{expected}]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one scenario and print its summary (optionally plot and export)."""
    scenario = get_scenario(args.scenario, seed=args.seed, centre_gain=args.centre_gain)
    result = simulate(scenario.sim, scenario.attack)
    print(
        f"Scenario: {scenario.title} ({scenario.key}), seed {scenario.sim.seed}, "
        f"{scenario.sim.loop_mode} loop, centre_gain {scenario.sim.centre_gain}"
    )
    print("Simulation result:")
    for name, value in summarise(result).items():
        print(f"  {name:<32} {value}")
    if args.plot:
        out = plot_result(result, scenario, PLOT_DIR / f"{scenario.key}_seed{scenario.sim.seed}.png")
        print(f"Plot saved to {out}")
    if args.csv:
        out = write_csv(result, Path(args.csv))
        print(f"CSV saved to {out}")
    return 0


def _fmt_drones(drones: set[int]) -> str:
    """Set of indices -> "Drone 02, Drone 06" (or "none")."""
    return ", ".join(drone_label(i) for i in sorted(drones)) or "none"


def _fmt_ratio(value: float) -> str:
    """0.5 -> "0.50"; NaN (undefined, e.g. precision with no alarms) -> "-"."""
    return "-" if value != value else f"{value:.2f}"  # WHY: NaN is the only value != itself


def _metrics_lines(m: RunMetrics, alarmed: set[int]) -> list[str]:
    """Human-readable lines describing one detector's RunMetrics."""
    if not m.is_attack_run:
        verdict = "false alarm" if m.clean_false_alarm else "no anomaly detected"
        return [f"    no attack in this run: {verdict}",
                f"    drones in alarm:   {_fmt_drones(alarmed)}"]
    delay = f"{m.detection_delay:.1f} s" if m.detection_delay is not None else "-"
    return [
        f"    detected:          {'yes' if m.detected else 'no'} (delay {delay})",
        f"    identified:        {_fmt_drones(alarmed)}",
        f"    false-alarm drones: {_fmt_drones(m.false_alarm_drones)}",
        f"    precision/recall:  {_fmt_ratio(m.id_precision)} / {_fmt_ratio(m.id_recall)}",
        f"    swarm-level alarm: {'yes' if m.swarm_level_detected else 'no'}",
    ]


def _pct(value: float) -> str:
    """0.25 -> "25%"."""
    return f"{value:.0%}"


def _progress_printer() -> Callable[[float], None]:
    """Progress callback that prints at most every 5 %.

    WHY only every 5 %: printing after every run would flood the terminal.
    """
    shown = {"last": -1}

    def show(fraction: float) -> None:
        step = int(fraction * 20)
        if step != shown["last"]:
            shown["last"] = step
            print(f"\r  progress {step * 5:3d} %", end="", flush=True)

    return show


def cmd_detect(args: argparse.Namespace) -> int:
    """Run one scenario through the pipeline: detectors, fusion and recovery."""
    scenario = get_scenario(args.scenario, seed=args.seed, centre_gain=args.centre_gain)
    calibration = load_calibration(scenario.sim)
    det_cfg = calibration.thresholds if calibration else DetectorConfig()
    source = (f"calibrated ({calibration.created_utc})" if calibration
              else "analytical defaults (run `calibrate` to measure them)")
    pipe = run_pipeline(scenario.sim, scenario.attack, det_cfg)
    result = pipe.sim_result
    print(f"Scenario: {scenario.title} ({scenario.key}), seed {scenario.sim.seed}")
    print(f"Attacked drones (truth, evaluation view): {_fmt_drones(result.attacked_drones())}")
    print(f"Thresholds: {source}")
    print("Simulation result:")
    for run in pipe.all_runs():
        metrics = compute_metrics(result, run)
        expected = scenario.expected.get(run.short, "-")
        print(f"  [{run.short}] {run.name}  (expected: {expected})")
        for line in _metrics_lines(metrics, run.alarm_drones(after=metrics.onset)):
            print(line)
    if args.ml:
        _print_ml_detection(pipe, scenario)
    key = pattern_key(pipe.runs)
    print(f"  Pattern: {key} - {describe_pattern(key)}")
    if pipe.recovery is not None:
        rec = pipe.recovery.summary()
        vec = rec["attack_vector"]
        estimate = (f"({vec[0]:.2f}, {vec[1]:.2f}) m east/north, |D| = {rec['attack_magnitude']:.2f} m"
                    if vec is not None else "not available")
        print(f"  Recovery ({rec['method']}, {rec['status']}): estimated swarm-wide GPS offset "
              f"{estimate}; outlier anchors: {_fmt_drones(set(rec['outlier_anchors']))}")
    return 0


def _print_ml_detection(pipe: PipelineResult, scenario: Scenario) -> None:
    """Score the experimental Detector D on this run, if a model has been trained.

    WHY imported here: scikit-learn is slow to import, and only this
    option and train-ml need it.
    """
    from detection.detector_d_ml import load_model, run_ml

    trained = load_model()
    if trained is None:
        print("  [D] Experimental ML: no model yet (run `python cli.py train-ml`)")
        return
    run = run_ml(pipe.sim_result.observed(), trained, pipe.det_cfg, scenario.sim)
    metrics = compute_metrics(pipe.sim_result, run)
    print(f"  [D] {run.name}  (EXPERIMENTAL, trained on simulated data only)")
    for line in _metrics_lines(metrics, run.alarm_drones(after=metrics.onset)):
        print(line)


def cmd_train_ml(args: argparse.Namespace) -> int:
    """Train the experimental Random Forest (Detector D) and save it with its SHA-256."""
    from detection.detector_d_ml import TRAINING_SEED_START, train_and_save

    seed_start = args.seed_start if args.seed_start is not None else TRAINING_SEED_START
    print(f"Simulating {args.runs} training runs (seeds {seed_start}-{seed_start + args.runs - 1}) "
          "and training the EXPERIMENTAL Random Forest...")
    result, path = train_and_save(args.runs, seed_start=seed_start, directory=Path(args.out),
                                  progress=_progress_printer())
    print()
    print(f"Split by run (F6): {len(result.train_groups)} train runs ({result.n_train} rows), "
          f"{len(result.test_groups)} test runs ({result.n_test} rows)")
    print("Simulation result, held-out test runs:")
    for name, value in result.metrics.items():
        print(f"  {name:<10} {_fmt_ratio(value)}")
    (tn, fp), (fn, tp) = result.confusion
    print(f"  confusion  TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    print(f"  clean-row false-positive rate: {_fmt_ratio(result.clean_false_positive_rate)}")
    print("Recall per attack type:")
    for name, value in result.per_type_recall.items():
        print(f"  {name:<16} {_pct(value)}")
    print("Feature importance:")
    for name, value in sorted(result.feature_importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {value:.3f}")
    digest = (Path(args.out) / "rf.sha256").read_text(encoding="utf-8").split()[0]
    print(f"Saved {path}  sha256 {digest}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sweep one parameter over evaluation seeds and print detection rates."""
    values = ([float(v) for v in args.values.split(",")] if args.values
              else list(builtin_values(args.param, args.scenario)))
    print(f"Sweeping {args.param} over {values} on {args.scenario}, "
          f"{args.seeds} seeds each, {args.workers} worker(s)...")
    df = sweep(args.param, values, args.scenario, args.seeds, workers=args.workers,
               progress=_progress_printer())
    print()
    table = summarise_sweep(df)
    print("Simulation result: detection rate per value and detector")
    print(table.pivot(index="value", columns="detector", values="detection_rate")
          .map(_pct).to_string())
    print("False-alarm rate (honest drone flagged, or alarm before onset):")
    print(table.pivot(index="value", columns="detector", values="false_alarm_rate")
          .map(_pct).to_string())
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Raw results saved to {args.csv}")
    return 0


def cmd_evasion(args: argparse.Namespace) -> int:
    """Find each detector's 50 % and 90 % detection thresholds (stealth envelope)."""
    limit = None if args.max_delay <= 0 else args.max_delay
    print(f"Stealth envelope for {args.scenario}: {args.seeds} seeds per evaluation, "
          f"detection must happen within {f'{limit:g} s' if limit else 'any time'}...")
    df = stealth_envelope(args.scenario, n_seeds=args.seeds, max_delay_s=limit,
                          workers=args.workers, progress=_progress_printer())
    print()
    print("Simulation result (smallest attack detected in at least the target share of runs):")
    for _, row in df.iterrows():
        value = ("not reliably detected in the search range" if pd.isna(row["threshold"])
                 else f"{row['threshold']:.3f}")
        print(f"  [{row['detector']}] {row['param']} for {row['target_rate']:.0%} detection: {value}")
    return 0


def cmd_propagation(args: argparse.Namespace) -> int:
    """Measure how one spoofed drone displaces the others, at centre_gain 0 and 0.5."""
    scenario = get_scenario(args.scenario)
    print(f"Propagation study on {scenario.key}'s swarm (its attack settings are ignored): "
          f"{args.magnitude} m spoof on each drone in turn, seed {args.seed}")
    for res in propagation_study(scenario.sim, magnitude=args.magnitude, seed=args.seed):
        off = res.matrix[~np.eye(len(res.matrix), dtype=bool)]
        top = np.argsort(res.influence)[::-1][:3]
        print(f"Simulation result, centre_gain = {res.centre_gain}:")
        print(f"  displacement of other drones: mean {off.mean():.3f} m, max {off.max():.3f} m")
        print(f"  spoofed drone dragged itself: mean {res.self_displacement.mean():.2f} m")
        print("  most influential (PageRank): " + ", ".join(
            f"{drone_label(int(i))} ({res.influence[i]:.3f})" for i in top))
        print(f"  min separation: clean {res.min_separation_baseline:.2f} m, "
              f"worst under attack {res.min_separation_attacked.min():.2f} m")
        events = [(i, e) for i, evs in res.near_collisions.items() for e in evs]
        print(f"  near-collision events (< 1 m): {len(events)}")
        for i, e in events:
            print(f"    spoofing {drone_label(i)}: {drone_label(e.i)} and {drone_label(e.j)} "
                  f"at {e.t:.1f} s, {e.distance:.2f} m apart")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate thresholds on clean runs, print the table and save JSON."""
    scenario = get_scenario(args.scenario)
    if scenario.attack.type != "none":
        # WHY allowed: calibration only uses the scenario's swarm settings
        # (SimConfig) and always flies clean; the attack is never simulated.
        print(f"Note: {scenario.key}'s attack ({scenario.attack.type}) is ignored; "
              "calibration flies its swarm configuration with no attack.")

    show = _progress_printer()
    print(f"Calibrating on {args.seeds} clean runs of {scenario.key} "
          f"(target false-alarm rate {args.target:.1%})...")
    result = calibrate(scenario.sim, n_seeds=args.seeds, include_manoeuvres=not args.no_manoeuvres,
                       target_fa=args.target, progress=show)
    print()
    table = result.table()
    for col in ("fa_rate", "ci_low", "ci_high"):
        table[col] = table[col].map(lambda v: f"{v:.1%}")
    print("Simulation result:")
    print(table.to_string(index=False))
    for note in result.notes:
        print(f"  note: {note}")
    print(f"Seeds {result.seed_start}-{result.seed_end}; saved to {save_calibration(result)}")
    return 0


# --------------------------------------------------------------------------
# Phase 4: investigation, verification, tamper test
# --------------------------------------------------------------------------
def cmd_investigate(args: argparse.Namespace) -> int:
    """Run, detect, fuse, recover; open an incident and build its signed evidence package."""
    scenario = get_scenario(args.scenario, seed=args.seed, centre_gain=args.centre_gain)
    pipe = run_pipeline(scenario.sim, scenario.attack, load_or_default(scenario.sim))
    print(f"Scenario: {scenario.title} ({scenario.key}), seed {scenario.sim.seed}")
    if not should_open_incident(pipe.runs) and not args.force:
        print("Simulation result: no anomaly detected; no incident opened (use --force to open one).")
        return 0
    storage = Storage(args.db)
    incident, is_new = open_incident(pipe, scenario.key, storage)
    existing = storage.get_package(incident.incident_id)
    if not is_new and existing is not None:
        print(f"This exact run (same configs and seed) was already investigated as {incident.incident_id}.")
        print(f"Package zip: {existing['zip_path']}")
        return 0
    paths = package_incident(incident, pipe, storage, out_root=args.out, keys_dir=args.keys)
    incident_id = incident.incident_id
    print(f"Incident {incident_id} opened (pattern {incident.pattern}).")
    print("Explanation (simulation result):")
    for sentence in incident.explanation:
        print(f"  {sentence}")
    print(f"Package folder:         {paths.folder}")
    print(f"Package zip:            {paths.zip_path}")
    print(f"PDF report:             {paths.report}")
    print(f"Manifest SHA-256:       {paths.manifest_sha256}")
    print(f"Previous manifest hash: {paths.previous_manifest_hash}")
    return 0


def _print_report(report: VerificationReport, indent: str = "  ") -> None:
    """Print every check as "STATUS  name - detail", then the overall result."""
    for c in report.checks:
        print(f"{indent}{c.status:<8}{c.name} - {c.detail}")
    print(f"{indent}OVERALL: {report.overall}")


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a package folder or zip; exit code 0 only if nothing FAILs."""
    storage = Storage(args.db)
    report = verify_package(args.path, storage=storage, trusted_key=Path(args.keys) / "public_key.pem")
    print(f"Verifying {args.path} (incident {report.incident_id})")
    _print_report(report)
    entry = storage.custody_entry(report.incident_id) if report.incident_id else None
    if entry is not None and report.manifest_sha256:
        storage.log_custody(report.incident_id, report.manifest_sha256, entry["previous_sha256"],
                            "verified" if report.overall == "PASS" else "verify_failed")
    print("A PASS shows the evidence is unchanged since signing; it does not prove the telemetry was genuine.")
    return 0 if report.overall == "PASS" else 1


def cmd_tamper_test(args: argparse.Namespace) -> int:
    """Flip one bit of one file in a COPY of the package and show that verification fails."""
    storage = Storage(args.db)
    trusted = Path(args.keys) / "public_key.pem"
    result = tamper_test(args.path, args.file, byte_index=args.byte, storage=storage, trusted_key=trusted)
    print(f"Tamper test on a temporary copy of {args.path}")
    print(f"  changed {result.target_file} byte {result.byte_index}: "
          f"0x{result.old_byte:02x} -> 0x{result.new_byte:02x} (one bit flipped)")
    print(f"  copy before tampering: {result.before.overall}")
    print("  copy after tampering:")
    _print_report(result.after, indent="    ")
    original = verify_package(args.path, storage=storage, trusted_key=trusted)
    print(f"  original package (never modified): {original.overall}")
    caught = result.before.overall == "PASS" and result.after.overall == "FAIL" and original.overall == "PASS"
    print("Tamper detected." if caught else "WARNING: tampering was NOT detected as expected.")
    return 0 if caught else 1


def _add_storage_args(parser: argparse.ArgumentParser) -> None:
    """--db and --keys options shared by the forensic commands."""
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--keys", default="keys", help="folder holding the Ed25519 key pair")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with its subcommands."""
    parser = argparse.ArgumentParser(prog="swarmguard", description="SwarmGuard command line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-scenarios", help="list preset scenarios").set_defaults(func=cmd_list_scenarios)
    run = sub.add_parser("run", help="simulate one scenario")
    run.add_argument("--scenario", required=True, help="scenario key (see list-scenarios)")
    run.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    run.add_argument(
        "--centre-gain", type=float, default=None, dest="centre_gain",
        help="override closed-loop centre_gain (0 = pure consensus formation keeping)",
    )
    run.add_argument("--plot", action="store_true", help="save a PNG to data/plots/")
    run.add_argument("--csv", default=None, help="write the run to this CSV file")
    run.set_defaults(func=cmd_run)

    detect = sub.add_parser("detect", help="run detectors A, B, C on one scenario")
    detect.add_argument("--scenario", required=True, help="scenario key (see list-scenarios)")
    detect.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    detect.add_argument("--centre-gain", type=float, default=None, dest="centre_gain",
                        help="override closed-loop centre_gain")
    detect.add_argument("--ml", action="store_true",
                        help="also score the experimental ML detector D (needs train-ml first)")
    detect.set_defaults(func=cmd_detect)

    ml = sub.add_parser("train-ml", help="train the experimental Random Forest detector D")
    ml.add_argument("--runs", type=int, default=300, help="simulated training runs (default 300)")
    ml.add_argument("--seed-start", type=int, default=None, dest="seed_start",
                    help="first training seed (default 100000, far from calibration/evaluation seeds)")
    ml.add_argument("--out", default=str(Path("data") / "models"), help="model output folder")
    ml.set_defaults(func=cmd_train_ml)

    cal = sub.add_parser("calibrate", help="calibrate detector thresholds on clean runs")
    cal.add_argument("--seeds", type=int, default=200, help="number of clean runs (default 200)")
    cal.add_argument("--scenario", default="clean_baseline", help="scenario whose swarm settings are calibrated (its attack is ignored)")
    cal.add_argument("--target", type=float, default=0.01, help="target false-alarm rate per run")
    cal.add_argument("--no-manoeuvres", action="store_true", dest="no_manoeuvres",
                     help="do not add legitimate manoeuvres to odd seeds")
    cal.set_defaults(func=cmd_calibrate)

    sw = sub.add_parser("sweep", help="detection rates while one parameter changes")
    sw.add_argument("--param", required=True,
                    help="magnitude, drift_rate, n_targets, anchor_ratio, n_drones or gps_sigma")
    sw.add_argument("--scenario", required=True, help="base scenario key")
    sw.add_argument("--values", default=None, help="comma-separated values (default: built-in list)")
    sw.add_argument("--seeds", type=int, default=50, help="evaluation seeds per value (from 5000)")
    sw.add_argument("--workers", type=int, default=1, help="parallel processes")
    sw.add_argument("--csv", default=None, help="save the raw per-seed results here")
    sw.set_defaults(func=cmd_sweep)

    ev = sub.add_parser("evasion", help="stealth envelope: smallest reliably detected attack")
    ev.add_argument("--scenario", required=True, help="attack scenario key")
    ev.add_argument("--seeds", type=int, default=40, help="seeds per evaluated value")
    ev.add_argument("--max-delay", type=float, default=10.0, dest="max_delay",
                    help="detections slower than this (s) do not count; 0 = no limit")
    ev.add_argument("--workers", type=int, default=1, help="parallel processes")
    ev.set_defaults(func=cmd_evasion)

    pr = sub.add_parser("propagation", help="how one spoofed drone disturbs the swarm")
    pr.add_argument("--scenario", default="clean_baseline", help="scenario whose swarm is used")
    pr.add_argument("--magnitude", type=float, default=6.0, help="spoof size in metres")
    pr.add_argument("--seed", type=int, default=5000, help="seed for both runs")
    pr.set_defaults(func=cmd_propagation)

    inv = sub.add_parser("investigate", help="open an incident and build a signed evidence package")
    inv.add_argument("--scenario", required=True, help="scenario key (see list-scenarios)")
    inv.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    inv.add_argument("--centre-gain", type=float, default=None, dest="centre_gain",
                     help="override closed-loop centre_gain")
    inv.add_argument("--force", action="store_true", help="open an incident even if nothing alarmed")
    inv.add_argument("--out", default=str(Path("data") / "packages"), help="package output folder")
    _add_storage_args(inv)
    inv.set_defaults(func=cmd_investigate)

    ver = sub.add_parser("verify", help="verify an evidence package (folder or zip)")
    ver.add_argument("path", help="package folder or .zip")
    _add_storage_args(ver)
    ver.set_defaults(func=cmd_verify)

    tam = sub.add_parser("tamper-test", help="flip one bit in a COPY of a package and re-verify")
    tam.add_argument("path", help="package folder or .zip (never modified)")
    tam.add_argument("--file", required=True, help="file inside the package, e.g. detections.csv")
    tam.add_argument("--byte", type=int, default=None, help="byte index to flip (default: random, seed 0)")
    _add_storage_args(tam)
    tam.set_defaults(func=cmd_tamper_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
