"""SwarmGuard command-line interface (Phase 1 commands).

Usage:
    python cli.py list-scenarios
    python cli.py run --scenario single_hijack [--seed 3] [--plot] [--csv out.csv]

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

from attacks.scenarios import Scenario, get_scenario, list_scenarios  # noqa: E402
from core.schema import SimResult  # noqa: E402
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
