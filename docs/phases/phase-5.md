############################################################################
# PHASE 5 — PREMIUM STREAMLIT APPLICATION
# Files: .streamlit/config.toml, assets/style.css, ui/theme.py, ui/state.py,
#        ui/components.py, ui/plots.py, app.py, views/*.py
############################################################################

DESIGN DIRECTION
"Mission control, not marketing page." Reference feel: Linear, the Vercel
dashboard, professional security-operations consoles. Calm, dense, precise.
The data is the decoration. If docs/design-refs/ contains screenshots,
study them and match their layout, spacing and hierarchy within Streamlit's
limits.

BANNED (these make apps look AI-generated)
- emojis anywhere in the UI; use Streamlit Material icons instead
  (":material/radar:", ":material/shield:", ":material/description:", ...)
- purple/blue gradients, glowing text, rainbow palettes
- default st.metric boxes for the main KPIs (use custom cards)
- Plotly's default white theme or default colour cycle
- centred long paragraphs; stacks of st.info/st.success boxes
- more than one accent colour

=== 5.1 ui/theme.py (single source of truth) ===
Colour tokens:
  bg #0A0D12, surface #11161D, surface_2 #161C25, border #1F2733,
  text #E6EAF0, text_muted #8A94A6, accent #3DD6C6 (the ONLY accent),
  status_ok #3FB950, status_warn #D29922, status_alarm #F85149,
  status_none #6E7681
Status colours are used only for status.
Detector colours, fixed everywhere: A #7AA2F7, B #3DD6C6, C #E0AF68, D #BB9AF7.
Typography: Inter (UI), JetBrains Mono (numbers, coordinates, hashes, IDs).
Scale: page title 22px/600; section label 12px/600 uppercase, letter-spacing
0.06em, text_muted; body 14px; data 13px mono with tabular numerals.
Spacing on an 8px grid; card radius 12px; 1px border hairlines.
Also exports the Plotly template (5.4) and status -> colour/label maps.

=== 5.2 .streamlit/config.toml and assets/style.css ===
config.toml: theme base dark, colours from the tokens, font sans serif,
client.toolbarMode = "minimal", hide the default sidebar navigation (we use
st.navigation).
style.css (injected once by app.py): Google Fonts import for Inter and
JetBrains Mono with system fallbacks; hide Streamlit header decoration,
main menu, footer and deploy button; max content width ~1400px with 24px
gutters; classes for .sg-card, .sg-kpi, .sg-pill-{status}, .sg-section,
.sg-feed, .sg-statusbar, .sg-inspector, .sg-mono; sidebar styling; button
styling (primary = accent fill, secondary = outline, at most one primary
per screen); dataframe header styling.

=== 5.3 ui/components.py (HTML via st.html or st.markdown + CSS classes) ===
  status_bar(scenario, seed, sim_time, overall_status, calibrated): top bar
    on every page: "SWARMGUARD", scenario, seed, time, status pill,
    calibration state.
  section(title, caption=None)
  kpi_card(label, value, delta=None, status=None): surface background,
    border, muted uppercase label, 28px mono value; kpi_row(items).
  status_pill(status): 10%-opacity status-colour background, solid
    status-colour text, always with a text label (never colour alone).
  drone_table(step_results): st.dataframe with column_config (progress
    columns for scores, formatted numbers, no index).
  event_feed(alerts): vertical timeline, mono timestamps, status dots.
  drone_inspector(drone, data): "properties pane": per-detector status and
    key numbers, fusion verdict, explainer reasons, recommended action
    text (e.g. "Do not trust GPS for Drone 04; navigate on velocity/anchor
    estimate; review incident SG-...").
  scenario_card(scenario): title, description, expected detector outcomes.
  verification_table(report): PASS/FAIL rows with icons and text.
  empty_state(message, action_label, on_click): one sentence + one button.
  explain_box(text): a compact "How to read this" expander for charts.

=== 5.4 ui/plots.py (Plotly, template "swarmguard") ===
Template: transparent paper, surface plot background, border-coloured
gridlines at low opacity, Inter font, muted axis labels, no in-figure titles
(titles come from section()), 1.5px lines, threshold lines dashed in
text_muted with right-aligned labels, legends horizontal above the plot.
Figures:
  swarm_map(...)  THE CENTREPIECE
    Real dark basemap of Pune with Plotly map traces and the
    "carto-darkmatter" style (no API key), centred on (18.5204, 73.8567),
    using enu_to_latlon. Drones as small circles coloured by status; faint
    trails (last 20 positions); anchored drones as outlined diamonds;
    formation links as 20%-opacity hairlines (toggle). Animation frames
    every 5 steps (120 frames for 60 s) with play/pause and a time slider.
    WHY stride 5: 600 frames make the browser lag.
    Evaluation view toggle: true positions as hollow markers joined to GPS
    by dashed lines, with a clear "Evaluation view: ground truth" badge.
    Offline fallback: if the "Offline map" toggle is on, draw the same data
    on a dark metric grid (needed if internet fails during the demo).
    Light-show mode: a second, larger low-opacity marker layer under each
    drone so formations glow and read clearly.
  detector_timeseries(runs, drone): A vote fraction (0.5 line); NIS on a log
    axis (9.21 line); CUSUM (h line); C residual (tau_c line); onset line
    labelled "simulation onset".
  residual_heatmap(run_a, step); comparison_bar(summary) with Wilson error
  bars; detection_curve(sweep_df, param) one line per detector with CI band;
  delay_box(sweep_df); stealth_envelope_chart(df); propagation_heatmap(M);
  pagerank_bar(scores); recovery_map(gps, recovered, anchors, inliers,
  truth=None); fusion_chart(fusion_df, drone) belief + plausibility;
  min_separation_chart(series).

=== 5.5 ui/state.py ===
One place for st.session_state keys: scenario_key, sim_cfg, atk_cfg, seed,
sim_result, detection_runs, fusion, recovery, incident_id.
get_or_run_pipeline(): simulate -> detect -> fuse -> recover, cached with
st.cache_data keyed on (config hashes, seed, detector config hash);
creates an incident and stores it if any detector alarms.

=== 5.6 app.py (router) ===
st.set_page_config(page_title="SwarmGuard", layout="wide",
page_icon=":material/radar:"); inject style.css; st.navigation with
sections and Material icons:
  Operations: Home, Dashboard, Swarm Simulator, Attack Lab
  Analysis: Detection Analysis, Experiments, Recovery
  Forensics: Incidents, Evidence Centre
  Reference: Methodology
Each st.Page points to a file in views/.

=== 5.7 views ===
home.py: product-style overview: one-line tagline, three short value
  statements (Detect / Recover / Prove), the architecture as a clean
  st.graphviz_chart styled with theme colours, and four quick-start cards
  (single_hijack, slow_drift, rigid_swarm_shift, light_show_pune) that run
  the pipeline and st.switch_page to the Dashboard.
dashboard.py: status bar; KPI row (total, normal, suspicious, compromised,
  no evidence); the swarm map with a step slider; drone table; event feed;
  latest incidents; a pattern banner when fusion detects one ("Coordinated
  swarm shift suspected").
simulator.py: scenario picker + seed; animated map; Evaluation view and
  Offline map toggles; min-separation chart; a short legend explaining
  slots, anchors and links.
attack_lab.py: left column: presets or manual controls in expanders
  (Swarm, Sensors, Anchors, Manoeuvres, Attack, Detector thresholds with a
  "use calibrated" checkbox); right column: one primary Run button, a
  results table (per detector: detected, delay, identified drones, false
  alarms), expected-vs-actual check against the scenario, and a link to
  the auto-created incident. Validation messages, never tracebacks.
detection.py: drone selector; detector timeseries; residual heatmap with
  step slider; detector comparison table; fusion chart; agreement matrix
  (drones × detectors, with labels); Drone Inspector in a right column.
  An explain_box under each chart.
experiments.py: tabs: Calibration (run, thresholds, achieved FA rates,
  Apply), Sweeps (param, scenario, seeds, workers; charts), Evasion
  (stealth envelope), Propagation (heatmap, PageRank, most influential
  drone, near-collision count). Runtime warning before long jobs; progress
  bars; cached results.
recovery.py: recovery map with inlier/outlier anchors; estimated attack
  vector; multilateration results per flagged drone, including
  INSUFFICIENT_NEIGHBOURS; before/after error table (labelled evaluation).
incidents.py: filterable table (date, pattern, affected drones); select ->
  timeline figure, explainer text, detector details; side-by-side compare
  of two incidents (delays, detectors that alarmed, offset size).
evidence.py: pick an incident -> build the package if missing; download
  buttons (zip, PDF, CSV); Verify Evidence -> verification table; Tamper
  Test on a copy with before/after side by side and the changed byte;
  custody chain table; public key fingerprint and PEM in an expander; a
  callout on what a signature does and does not prove.
methodology.py: st.latex for every equation (pairwise residual, majority
  vote, Kalman predict/update, NIS, CUSUM, anchor residual + median
  example, Dempster's rule, MDS double centering, Kabsch, multilateration
  residual, Wilson interval); assumptions; blind-spot table; fixes F1–F8;
  credits R6; limitations.

EMPTY AND ERROR STATES
Every view works on a fresh database: show empty_state(), never a blank page
or a traceback.

MOTION
Only meaningful motion: the swarm animation and progress bars.

QUALITY BAR
Check every view at about 1440px and 1024px wide; fix spacing, alignment and
overflow. Consistency beats decoration.

ACCEPTANCE CHECKS
  .venv\Scripts\streamlit run app.py
  every view loads on a fresh database with no exceptions
  the light_show_pune quick start runs end to end, shows the map over Pune,
    and produces an incident whose evidence package verifies in the
    Evidence Centre
############################################################################