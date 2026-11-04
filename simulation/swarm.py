"""Step-by-step swarm simulator.

SwarmSimulator(sim_config, attack_config).run() -> SimResult

Each step k (time t = k * dt):
  1. Planned swarm centre c(t) and its velocity v_c (finite difference).
  2. Formation offsets o_i(t) (shape morphing + manoeuvres); slots = c + o_i.
  3. GPS receiver noise, then the attack offset (rigid_rotation needs the
     clean noisy readings first).
  4. All sensors read the CURRENT true state.
  5. Control computes a commanded velocity (open or closed loop).
  6. v = clip_norm(v_cmd, max_speed); stored as the true velocity flown
     during [t, t + dt]; the velocity sensor measures this v.
  7. The Frame is stored, then true_pos += v * dt.

Control laws (kp = formation gain, kc = centre_gain, both in 1/s):
  open loop:    v_cmd_i = kp * (slot_i - true_i) + v_c
      WHY: spoofing corrupts only measurements and never moves the drones,
      which gives the cleanest test of the detectors.
  closed loop:  c_est   = mean_j(gps_j) - mean_j(o_j)
                v_cmd_i = kp * (c_est + o_i - gps_i) + kc * (c(t) - c_est) + v_c
      WHY: real drones steer by what they measure. The first term keeps
      the formation using GPS. Because c_est is a consensus of everyone's
      GPS, one spoofed drone moves every drone's target by offset / N: this
      is how an attack on one drone propagates to the swarm (SwarmFuzz
      idea). The second term pulls the GPS-estimated centre onto the
      planned path, so a swarm-wide GPS drift drags the TRUE swarm off
      course while the formation still looks perfect. With kc = 0 this
      reduces to pure formation keeping.

Why the pure consensus law (kc = 0) ignores a swarm-wide spoof:
  1. The formation term never moves the swarm's average. Summing it over
     all N drones:
         sum_i kp * (c_est + o_i - gps_i) = kp * (N c_est + sum o - sum gps) = 0
     because c_est is defined as mean(gps) - mean(o). Only v_c moves the
     average, and v_c comes from the plan, not from GPS.
  2. A swarm-wide offset D cancels inside every drone's command:
         gps_i -> gps_i + D   and   c_est -> c_est + D
         (c_est + D) + o_i - (gps_i + D) = c_est + o_i - gps_i
     The commands are exactly the same as without the attack, so the true
     swarm keeps flying the plan. Only the reported GPS is wrong. This is
     the same cancellation that blinds Detector A.
  The path term kc * (c(t) - c_est) breaks the cancellation. c_est now
  contains +D, so the whole swarm is steered by -D until c_est = c(t),
  which means the true centre ends at c(t) - D: dragged off course.

Single-drone spoof (single_hijack, D = 10 m east on Drone 04, N = 10),
measured against a clean run with the same seed:
  kc = 0    c_est moves by D/N = 1 m. Every other drone ends up
            permanently 1.0 m east of where it should be (the SwarmFuzz
            propagation effect). Drone 04 ends about 9 m west.
  kc = 0.5  (default) the path term pulls c_est back onto the plan, so the
            other drones are nudged by at most about 0.35 m and then return.
            Drone 04 is dragged the full 10 m west.
Set centre_gain per scenario (get_scenario(..., centre_gain=0.0), or the
CLI's --centre-gain) to compare both behaviours.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from attacks.gps_attacks import AttackModel
from config import AttackConfig, SimConfig
from core.schema import Frame, ObservedFrame, SimResult
from core.utils import clip_norm, round_half_up
from simulation.formation import ShapeSchedule, apply_manoeuvres
from simulation.sensors import RngBundle, SensorSuite

logger = logging.getLogger(__name__)

# Standard deviation (m) of the random start-position jitter.
INITIAL_JITTER_M: float = 0.2


def centre_path(config: SimConfig, times: np.ndarray) -> np.ndarray:
    """Planned swarm centre c(t) for every time in `times`, shape (T, 2).

    straight:  c(t) = speed * t * (cos h, sin h)
    loop:      a circle of radius R travelled counter-clockwise at angular
               speed w = speed / R, starting at the origin heading h:
               C    = R * (-sin h, cos h)              (circle centre)
               c(t) = C + R * (cos(h - pi/2 + w t), sin(h - pi/2 + w t))
    waypoints: straight lines between waypoints at constant speed,
               starting at the first waypoint and hovering at the last.
    """
    h = math.radians(config.heading_deg)
    speed = config.center_speed
    if config.path == "straight":
        return speed * times[:, None] * np.array([math.cos(h), math.sin(h)])
    if config.path == "loop":
        radius = config.loop_radius
        omega = speed / radius
        centre = radius * np.array([-math.sin(h), math.cos(h)])
        phase = h - math.pi / 2 + omega * times
        return centre + radius * np.column_stack([np.cos(phase), np.sin(phase)])
    points = np.asarray(config.waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    travelled = np.minimum(speed * times, cum[-1])
    x = np.interp(travelled, cum, points[:, 0])
    y = np.interp(travelled, cum, points[:, 1])
    return np.column_stack([x, y])


class SwarmSimulator:
    """Runs one deterministic simulation for a (SimConfig, AttackConfig) pair."""

    def __init__(self, sim_config: SimConfig, attack_config: AttackConfig | None = None) -> None:
        self.config = sim_config
        self.attack_config = attack_config if attack_config is not None else AttackConfig()
        self.attack = AttackModel(self.attack_config, sim_config.n_drones)
        # WHY SeedSequence.spawn: one seed gives independent child streams
        # for each random part of the run (rule R2). Changing e.g. how
        # anchors are chosen never alters the GPS noise.
        jitter_ss, anchor_choice_ss, sensor_ss = np.random.SeedSequence(sim_config.seed).spawn(3)
        self._jitter_rng = np.random.default_rng(jitter_ss)
        self._anchor_choice_rng = np.random.default_rng(anchor_choice_ss)
        self.sensors = SensorSuite(sim_config, RngBundle.from_seed_sequence(sensor_ss))
        self.schedule = ShapeSchedule(
            sim_config.formation,
            sim_config.n_drones,
            sim_config.radius,
            sim_config.shape_sequence,
            sim_config.custom_points or None,
        )
        self.anchored, self.faulty = self._choose_anchors()

    def _choose_anchors(self) -> tuple[np.ndarray, np.ndarray]:
        """Pick which drones carry anchors, and which of those anchors are faulty.

        n_anchored = round(anchor_ratio * N), n_faulty = round(anchor_fault_ratio
        * n_anchored), both rounding half up. Chosen once per run.
        """
        n = self.config.n_drones
        rng = self._anchor_choice_rng
        n_anchored = round_half_up(self.config.anchor_ratio * n)
        anchored_idx = rng.choice(n, size=n_anchored, replace=False)
        n_faulty = round_half_up(self.config.anchor_fault_ratio * n_anchored)
        faulty_idx = rng.choice(anchored_idx, size=n_faulty, replace=False) if n_faulty else []
        anchored = np.zeros(n, dtype=bool)
        anchored[anchored_idx] = True
        faulty = np.zeros(n, dtype=bool)
        faulty[faulty_idx] = True
        return anchored, faulty

    def formation_offsets(self, t: float) -> np.ndarray:
        """Formation offsets at time t: shape schedule, then manoeuvres."""
        return apply_manoeuvres(self.schedule.offsets(t), t, self.config.manoeuvres)

    def _command(
        self,
        true_pos: np.ndarray,
        gps: np.ndarray,
        offsets: np.ndarray,
        centre: np.ndarray,
        v_centre: np.ndarray,
    ) -> np.ndarray:
        """Commanded velocity for every drone (see the module docstring)."""
        cfg = self.config
        if cfg.loop_mode == "open":
            return cfg.kp * (centre + offsets - true_pos) + v_centre
        centre_est = gps.mean(axis=0) - offsets.mean(axis=0)
        formation_term = cfg.kp * (centre_est + offsets - gps)
        path_term = cfg.centre_gain * (centre - centre_est)
        return formation_term + path_term + v_centre

    def run(self) -> SimResult:
        """Simulate the whole flight and return every Frame."""
        cfg = self.config
        n_steps = cfg.n_steps
        times = np.arange(n_steps + 1) * cfg.dt
        centres = centre_path(cfg, times)
        # WHY forward difference: v_c[k] is the velocity that carries the
        # planned centre from c(t_k) to c(t_k+1), matching one Euler step.
        v_centres = np.diff(centres, axis=0) / cfg.dt

        true_pos = centres[0] + self.formation_offsets(0.0)
        true_pos = true_pos + self._jitter_rng.normal(0.0, INITIAL_JITTER_M, size=true_pos.shape)
        frames: list[Frame] = []
        for k in range(n_steps):
            t = float(times[k])
            offsets = self.formation_offsets(t)
            slots = centres[k] + offsets
            noise = self.sensors.gps_noise()
            attack_offset, labels = self.attack.offsets(t, true_pos + noise)
            gps = self.sensors.gps(true_pos, attack_offset, noise)
            ranges, range_mask = self.sensors.ranges(true_pos)
            anchor_pos, anchor_mask = self.sensors.anchors(true_pos, self.anchored, self.faulty)

            v_cmd = self._command(true_pos, gps, offsets, centres[k], v_centres[k])
            velocity = clip_norm(v_cmd, cfg.max_speed)
            observed = ObservedFrame(
                step=k, t=t, gps_pos=gps, vel_meas=self.sensors.velocity(velocity),
                ranges=ranges, range_mask=range_mask,
                anchor_pos=anchor_pos, anchor_mask=anchor_mask,
            )
            frames.append(Frame(
                observed=observed, true_pos=true_pos, true_vel=velocity,
                attack_offset=attack_offset, attack_labels=labels,
                faulty_anchor=self.faulty, slot_pos=slots, anchored=self.anchored,
            ))
            true_pos = true_pos + velocity * cfg.dt
        logger.info(
            "simulated %d steps x %d drones (attack=%s, seed=%d)",
            n_steps, cfg.n_drones, self.attack_config.type, cfg.seed,
        )
        return SimResult(config=cfg, attack=self.attack_config, frames=frames)


def simulate(sim_config: SimConfig, attack_config: AttackConfig | None = None) -> SimResult:
    """Convenience function: SwarmSimulator(sim_config, attack_config).run()."""
    return SwarmSimulator(sim_config, attack_config).run()
