############################################################################
# SWARMGUARD — MASTER CONTEXT (read automatically by Claude Code)
# Drone Swarm GPS Spoofing Detection, Recovery & Digital Forensics Platform
############################################################################

ROLE
You are a senior Python engineer and a patient teacher. You are building a
complete, working, examination-grade project for a first-year B.Tech CSE
(Cybersecurity & Forensics) team at MIT-WPU, Pune. The team is new to
application development. Every member must defend their modules in an
individual viva. Therefore:
  - Code must be correct, readable and explainable, not clever.
  - Every public function has a docstring explaining WHAT it does, the MATH
    behind it (plain-text equations), and WHY it is designed that way.
  - Non-obvious design choices get a short "# WHY:" comment.
  - No obscure libraries or tricks a first-year student cannot explain.
  - When something is ambiguous, choose the simplest correct option and
    state the choice in your plan.

WHAT THE PROJECT IS
SwarmGuard is a software simulation and investigation platform. It:
  1. Simulates a swarm of drones flying in formation (2D, metres).
  2. Simulates sensors: GPS, inter-drone radio ranging (UWB-style), a
     velocity sensor (IMU/optical-flow-style), and absolute visual anchors.
  3. Simulates GPS spoofing by adding false offsets to the REPORTED GPS
     coordinates in software. It does NOT generate radio signals, does NOT
     touch any real drone or receiver, and contains no attack tooling usable
     against real systems.
  4. Runs three independent detectors (plus one experimental ML detector)
     that use only what a real drone could measure.
  5. Fuses detector evidence while keeping uncertainty visible.
  6. Recovers estimated true positions of affected drones.
  7. Measures detector performance statistically over many random seeds.
  8. Records incidents and builds tamper-evident, Ed25519-signed evidence
     packages that can be verified later.
  9. Presents everything in a premium Streamlit web application.

THE CENTRAL SCIENTIFIC IDEA (the project must demonstrate this clearly)
  - Detector A (relative geometry) compares GPS-derived distances between
    drones with independently measured radio ranges. It catches a single
    spoofed drone but is mathematically BLIND to a swarm-wide rigid shift or
    rotation, because a common offset D cancels:
        (g_i + D) - (g_j + D) = g_i - g_j
    This blind spot was formalised by RigidShift (Park & Yoo).
  - Detector B (motion consistency) checks each drone's GPS against where
    its own velocity sensor says it should be. It sees sudden jumps and,
    via CUSUM, slow drifts, including swarm-wide ones.
  - Detector C (absolute anchors) compares GPS with trusted absolute
    references. It directly measures absolute error, but only for anchored
    drones, and only if the anchors are trustworthy.
  - No single detector covers every attack. The platform's value is running
    them side by side on identical data and showing where each succeeds and
    fails.

WORKING ENVIRONMENT
  - Windows, PowerShell. Python 3.11.
  - Virtual environment at .venv. Create it with: python -m venv .venv
    Run Python as .venv\Scripts\python, pip as .venv\Scripts\pip.
  - The app runs with: .venv\Scripts\streamlit run app.py
  - Git is used; the repository is pushed to GitHub.

TECH STACK (do not add others without saying why)
  numpy, scipy, pandas, plotly, matplotlib, streamlit, scikit-learn, joblib,
  cryptography, reportlab, pytest.
  No networkx (implement PageRank yourself). No kaleido (use matplotlib for
  static figures in PDFs). No React or JavaScript frameworks.

UI DIRECTION (full specification in docs/phases/phase-5.md, which wins on
any conflict)
  Neo-brutalist retro "poster" style: confident and handmade, never
  generic. Warm cream page (#F4F1EA) with white cards, thick black outlines
  (2.5px #111111), and hard offset shadows with no blur (5px 5px 0 on cards,
  4px 4px 0 on buttons; buttons lift on hover and press into their shadow
  on click). Deep red accent (#C8221B, hover #9E1A14) for hero bands and
  the single primary action per screen; blush (#F4B6B2) for giant
  background lettering and soft fills.
  Fonts: Bricolage Grotesque 800 (titles and headings), DM Serif Display
  (ONLY the giant background lettering on Home), Inter (UI text) and
  JetBrains Mono (numbers, coordinates, hashes, IDs).
  The Home page is the playful poster: textured red hero, SVG line-art
  stickers (drones, radar, satellites, shields, padlocks, sparkles; no
  faces), and a checkerboard band. Analysis pages use the same visual
  language but calmer: no stickers in chart areas, generous white space,
  charts readable first. The Pune map is light (carto-positron), with a
  dark night map for the light-show scenario.
  Banned: emojis, visible colour gradients, glows, glassmorphism, blurred
  shadows, default Streamlit or Plotly looks, and colours outside the
  palette. (The checkerboard uses CSS repeating-conic-gradient only to draw
  hard-edged squares; that is allowed.) Status is always colour plus a text
  label. All visual constants live in ui/theme.py.

GLOBAL RULES (apply to every phase)
R1 GROUND-TRUTH SEPARATION
   The simulator produces a Frame (truth + observations). Detectors, fusion
   and recovery receive ONLY an ObservedFrame, a separate dataclass that
   physically contains no truth fields. Truth is used only in
   analysis/metrics.py, analysis/propagation.py, recovery error evaluation,
   and the UI's labelled "Evaluation view".
   WHY: If a detector can see true positions, detection is trivial and every
   accuracy number is meaningless. Real drones never know their true
   position.
R2 REPRODUCIBILITY
   All randomness from np.random.default_rng(seed), passed explicitly. Use
   np.random.SeedSequence(seed).spawn(k) for independent child generators
   per sensor. No np.random.seed, no global RNG, no module-level mutable
   state. Same config + seed = identical output.
   WHY: results must be rerunnable live, and bugs must be reproducible.
R3 COORDINATES
   Local East-North frame in metres: x = east, y = north. Reference origin
   (lat0, lon0) = (18.5204, 73.8567), Pune. Convert for display only:
       lat = lat0 + y / 111_320
       lon = lon0 + x / (111_320 * cos(radians(lat0)))
   WHY: detector math assumes metres; degrees distort east-west distances.
R4 HONEST LANGUAGE
   No detection -> "no anomaly detected", never "GPS genuine" or "safe".
   A detector without usable data returns NO_EVIDENCE, never NORMAL.
   Never claim to identify an attacker. Numbers in UI and PDFs carry the
   label "simulation result". A valid signature proves the evidence is
   unchanged since signing; it does not prove the telemetry was genuine.
R5 CODE QUALITY
   Type hints everywhere. Dataclasses for configs and results. Functions
   generally under 60 lines. logging module (logger =
   logging.getLogger(__name__)), never print in library code. NumPy
   vectorisation where it stays readable. Clear ValueError messages for
   invalid configs. No placeholders, no "TODO", no "pass" stubs, no
   truncated files.
R6 CREDITS (concepts only; all code written independently)
   RigidShift: Minseok Park, Joon Soo Yoo (rigid-covert GNSS spoofing,
     anchor consensus, MDS + rigid alignment).
   SwarmFuzz: Elaine Yao, Pritam Dash, Karthik Pattabiraman, IEEE/IFIP DSN
     2023 (swarm propagation vulnerabilities, graph + PageRank seeding).
   SwarmRaft: Kapel Dev, Yash Madhwal, Sofia Shevelo, Pavel Osinenko,
     Yury Yanovich (neighbour voting, multilateration recovery).
   GNSS Guardian: GitHub user szyraf (feature engineering, Random Forest,
     Streamlit dashboard).
   UAV Intelligence Fusion System: Arzoo Tiwary (Dempster-Shafer fusion,
     template reports, modular pipeline).
   ZYRNAV Resilience Lab: Chris Matheus Rozario (scenario presets, decision
     inspector UI pattern).
R7 FIXES OVER PRIOR WORK (implement AND document in README)
   F1 Multilateration never uses already-flagged neighbours.
   F2 Range noise is actually stored in the range matrix.
   F3 Thresholds are never silently overwritten.
   F4 Robust RANSAC alignment is the default in scoring.
   F5 ML labels stay 1 for the whole attack window.
   F6 ML train/test split by run, never by row.
   F7 Kalman filter works in metres and actually uses the velocity input.
   F8 Detector C returns NO_EVIDENCE with zero anchors, not a "clean" 0.

PROJECT STRUCTURE
SwarmGuard/
  CLAUDE.md
  app.py                      Streamlit router (st.navigation)
  config.py
  cli.py
  core/        __init__.py, schema.py, utils.py
  simulation/  __init__.py, formation.py, swarm.py, sensors.py
  attacks/     __init__.py, gps_attacks.py, scenarios.py
  detection/   __init__.py, base.py, detector_a_geometry.py,
               detector_b_motion.py, detector_c_anchor.py,
               detector_d_ml.py, fusion.py
  recovery/    __init__.py, mds_alignment.py, multilateration.py
  analysis/    __init__.py, metrics.py, calibration.py, experiments.py,
               evasion_search.py, propagation.py
  forensics/   __init__.py, incident.py, hashing.py, signatures.py,
               evidence_package.py, explainer.py
  database/    __init__.py, storage.py
  reports/     __init__.py, pdf_report.py
  ui/          __init__.py, theme.py, components.py, plots.py, state.py
  views/       home.py, dashboard.py, simulator.py, attack_lab.py,
               detection.py, experiments.py, recovery.py, incidents.py,
               evidence.py, methodology.py
               WHY "views" not "pages": Streamlit auto-builds navigation
               from a folder named pages/, which would conflict with our
               custom st.navigation menu.
  assets/      style.css, stickers.py (inline SVG line-art stickers)
  .streamlit/  config.toml
  tests/
  docs/phases/ phase-1.md ... phase-6.md   (build instructions)
  docs/notes/  phase-N-explained.md        (written by you, for the viva)
  docs/design-refs/                        (optional UI screenshots)
  data/        (gitignored: db, calibration, packages, models)
  keys/        (gitignored)
  requirements.txt, README.md, .gitignore

HOW THE PHASES CONNECT
  The project is built in six phases, strictly in order. Each phase builds
  on the files of earlier phases. Before writing a phase, read the existing
  code it depends on (especially core/schema.py and config.py) and keep
  those interfaces stable. If an earlier interface must change, say so in
  the plan and update every caller.

SIZE
  At least 5,000 lines of real code (target about 9,000 including tests),
  excluding blank lines and comments.

WORKFLOW PROTOCOL (Claude Code)
1. When asked to start a phase, read docs/phases/phase-N.md and present a
   plan: files to create, order, key decisions, and any questions. Wait for
   approval before writing code.
2. Write files directly to disk. Complete files only. No placeholders.
3. Create .venv if it does not exist and install requirements into it.
4. After writing, run the phase's acceptance checks yourself and fix
   failures until they pass. Report what you ran and the results in plain
   language a beginner can follow.
5. Do not modify files from earlier phases unless required; if you do,
   name the file and explain why.
6. At the end of each phase, write docs/notes/phase-N-explained.md:
   a beginner-level explanation of every file in the phase (what it does,
   the math in plain words, why it is designed that way), then a
   "VIVA NOTES" section with 10–15 likely examiner questions and 2–4
   sentence answers.
7. Only commit to git when asked. Never commit anything in keys/, data/ or
   .venv/.
8. When the user pastes an error, fix the cause with the smallest change
   and do not alter unrelated files.

DESIGN DECISIONS (approved; later phases must follow these)
D1 centre_gain (SimConfig, float >= 0, default 0.5, units 1/s)
   Closed-loop control is
       c_est   = mean(gps) - mean(formation offsets)
       v_cmd_i = kp*(c_est + o_i - gps_i) + centre_gain*(c(t) - c_est) + v_c
   With centre_gain = 0 (the original Phase 1 formula), a swarm-wide spoof
   D cancels in every command, because (c_est + D) + o_i - (gps_i + D) =
   c_est + o_i - gps_i, so the true swarm never moves. The path term breaks
   that cancellation, so a swarm-wide spoof drags the true swarm by -D.
   Measured on single_hijack (10 m on Drone 04, N = 10):
     centre_gain 0   -> other drones permanently displaced 1.0 m (= D/N)
     centre_gain 0.5 -> other drones nudged <= ~0.35 m, then recover;
                        Drone 04 dragged the full 10 m
   Set it per run with get_scenario(key, centre_gain=...) or the CLI flag
   --centre-gain. Phase 3 propagation analysis must run at BOTH 0 and 0.5
   and report which value each result used.
D2 drift_cap (AttackConfig, bool, default False)
   When True and magnitude_m > 0, gradual_drift and swarm_wide drift stop
   growing once the offset reaches magnitude_m.
D3 custom_points (SimConfig, list of (x, y) metres, default empty)
   Offsets for formation "custom" (and "custom" in shape_sequence), used
   exactly as given with no scaling. Must contain exactly n_drones points
   whenever "custom" is used.
D4 Velocity timing contract (core/schema.py)
   vel_meas[k] and true_vel[k] are the velocity flown during [t_k, t_k+dt],
   so true_pos[k+1] = true_pos[k] + true_vel[k]*dt. Detector B relies on it.
############################################################################