"""SwarmGuard command-line interface.

Usage:
    python cli.py list-scenarios
    python cli.py run --scenario single_hijack [--seed 3] [--plot] [--csv out.csv]
    python cli.py detect --scenario rigid_swarm_shift [--seed 5000]
    python cli.py calibrate [--seeds 200] [--scenario clean_baseline] [--target 0.01]

All numbers printed are simulation results, not measurements of real drones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

# WHY "Agg": draws straight to a PNG file without opening a window, so the
# command also works on machines without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402

from analysis.calibration import calibrate, load_calibration, save_calibration  # noqa: E402
from analysis.metrics import RunMetrics, compute_metrics  # noqa: E402
from attacks.scenarios import Scenario, get_scenario, list_scenarios  # noqa: E402
from config import DetectorConfig  # noqa: E402
from core.schema import SimResult  # noqa: E402
from detection import build_detectors  # noqa: E402
from simulation.swarm import simulate  # noqa: E402

logger = logging.getLogger(__name__)

PLOT_DIR = Path("data") / "plots"


def drone_label(index: int) -> str:
    """0-based index -> display name, e.g. 3 -> "Drone 04"."""
    return f"Drone {index + 1:02d}"


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


def cmd_detect(args: argparse.Namespace) -> int:
    """Run one scenario through detectors A, B and C and print their metrics."""
    scenario = get_scenario(args.scenario, seed=args.seed, centre_gain=args.centre_gain)
    result = simulate(scenario.sim, scenario.attack)
    calibration = load_calibration(scenario.sim)
    det_cfg = calibration.thresholds if calibration else DetectorConfig()
    source = (f"calibrated ({calibration.created_utc})" if calibration
              else "analytical defaults (run `calibrate` to measure them)")
    print(f"Scenario: {scenario.title} ({scenario.key}), seed {scenario.sim.seed}")
    print(f"Attacked drones (truth, evaluation view): {_fmt_drones(result.attacked_drones())}")
    print(f"Thresholds: {source}")
    print("Simulation result:")
    for detector in build_detectors(det_cfg, scenario.sim):
        run = detector.run(result.observed())
        metrics = compute_metrics(result, run)
        expected = scenario.expected.get(detector.short, "-")
        print(f"  [{detector.short}] {detector.name}  (expected: {expected})")
        onset = metrics.onset
        for line in _metrics_lines(metrics, run.alarm_drones(after=onset)):
            print(line)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate thresholds on clean runs, print the table and save JSON."""
    scenario = get_scenario(args.scenario)
    if scenario.attack.type != "none":
        # WHY allowed: calibration only uses the scenario's swarm settings
        # (SimConfig) and always flies clean; the attack is never simulated.
        print(f"Note: {scenario.key}'s attack ({scenario.attack.type}) is ignored; "
              "calibration flies its swarm configuration with no attack.")

    shown = {"last": -1}

    def show(fraction: float) -> None:
        # WHY only every 5 %: printing after every run would flood the terminal.
        step = int(fraction * 20)
        if step != shown["last"]:
            shown["last"] = step
            print(f"\r  progress {step * 5:3d} %", end="", flush=True)

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
    detect.set_defaults(func=cmd_detect)

    cal = sub.add_parser("calibrate", help="calibrate detector thresholds on clean runs")
    cal.add_argument("--seeds", type=int, default=200, help="number of clean runs (default 200)")
    cal.add_argument("--scenario", default="clean_baseline", help="scenario whose swarm settings are calibrated (its attack is ignored)")
    cal.add_argument("--target", type=float, default=0.01, help="target false-alarm rate per run")
    cal.add_argument("--no-manoeuvres", action="store_true", dest="no_manoeuvres",
                     help="do not add legitimate manoeuvres to odd seeds")
    cal.set_defaults(func=cmd_calibrate)
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
