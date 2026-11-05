############################################################################
# PHASE 5 — STREAMLIT APPLICATION: "IVORY + CONSOLE" PROFESSIONAL DESIGN
# Files: .streamlit/config.toml, assets/style.css, assets/stickers.py,
#        assets/hero.html (or ui/hero.py), ui/theme.py, ui/state.py,
#        ui/components.py, ui/plots.py, app.py, views/*.py,
#        plus updates to the shared palette module created in Phase 4
############################################################################

STEP 0 — BEFORE ANY UI CODE
1. This file is the final UI design. It REPLACES the neo-brutalist poster
   design. Update the UI DIRECTION section of CLAUDE.md to summarise this
   design (ivory app + dark green-black console panels, deep green brand,
   lavender tactile buttons, Instrument Serif / Bricolage Grotesque /
   Inter / JetBrains Mono, subtle purposeful motion). Also update
   PROJECT STRUCTURE if files differ. Commit to main.
2. Phase 4 created a shared palette module used by the PDF report. Update
   it to the palette below (brand = deep green, alarm = red) so the PDF and
   the app match. ui/theme.py must import from that module, so there is
   ONE source of truth for colours. Regenerate one example PDF and check it.

DESIGN DIRECTION
"A calm, premium security product." Two references are blended:
- Wispr Flow (wisprflow.ai): warm ivory canvas, deep green brand colour,
  soft lavender buttons, an elegant serif headline with an italic second
  line, small letter-spaced eyebrow labels, a floating pill navbar, and
  flowing motion (text gliding along curved paths, a waveform pill, a
  moving black ribbon of text).
- Urban Macaw (urbanmacaw.com), its STYLE only, never its imagery:
  tactile buttons with a dark outline and hard offset shadow that press
  into the shadow on click, crisply outlined cards, bold characterful
  numerals, and thick-outlined line-art illustrations.
The project is a cybersecurity tool, so operational content (live map,
telemetry, event feed, evidence verification) lives in dark green-black
CONSOLE PANELS inside the light ivory page. This reads as a professional
operations console without the clichés of AI dark mode.

Reference products for the overall level of polish: Stripe, Linear,
Vercel, Wispr Flow. Every screen should look intentional and finished.

BANNED
- neon, glow effects, purple/pink "cyber" gradients, glassmorphism,
  blurred drop shadows, rainbow palettes
- emojis anywhere (use our SVG line-art or Streamlit Material icons)
- red used for anything except alarms/destructive actions
- default Streamlit or Plotly styling left visible anywhere
- cartoon faces; illustrations are technical subjects only
- decorative motion that delays the user or loops distractingly outside
  the Home hero

=== 5.1 PALETTE (shared palette module; ui/theme.py re-exports it) ===
Light page:
  ivory      #FFFEF2   page background (Wispr)
  paper      #FFFFFF   cards
  ink        #111111   text, outlines, hard shadows (Urban Macaw)
  ink_2      #3D3D3A   secondary text
  muted      #77756E   captions, axis labels
  line       #E4E1D3   hairlines inside cards
  forest     #0B4D40   brand colour: links, active nav, key headings
  forest_2   #0F6B58   hover state of forest elements
  lavender   #E6DDFE   primary button fill (Wispr)
  lavender_2 #D7C9FD   button hover / pressed fill, selected states
  blush      #F4B6B2   rare soft highlight (Urban Macaw), e.g. heatmap mid
Console panels (dark):
  console    #0E1F1B   panel background (deep green-black)
  console_2  #132A25   raised rows inside panels
  console_ln #24413A   panel borders and gridlines
  console_tx #E9F1EC   panel text
  console_mu #8FA79E   panel secondary text
  mint       #7FD8B8   flight paths and positive signals on dark
Status (always paired with a text label, never colour alone):
  NORMAL #2F9E6B · SUSPICIOUS #E0A100 · ALARM / COMPROMISED #C8221B ·
  NO_EVIDENCE / INSUFFICIENT_EVIDENCE #8A8577
Detector colours (fixed everywhere):
  A #4C6EF5 (blue) · B #0CA5A5 (teal) · C #B7791F (ochre) · D #8C6FD8 (violet)
WHY: one brand colour (forest), one interaction colour (lavender), and red
reserved for real alarms, so alarms are impossible to miss.

=== 5.2 TYPOGRAPHY (Google Fonts with system fallbacks) ===
  Instrument Serif   page titles and hero headlines; italic for emphasis
                     ("Detect the drift, / *prove the attack.*")
  Bricolage Grotesque 600–800   KPI numbers, card titles, section labels
  Inter 400/500/600  body text, menus, tables, buttons
  JetBrains Mono     coordinates, hashes, IDs, timestamps, code
Scale: hero 64px serif (40px under 900px width); page title 40px serif;
section title 22px Bricolage 700; eyebrow 11px Inter 600 uppercase,
letter-spacing 0.14em, muted; body 15px/1.6 Inter; table 14px; data 13px
mono with tabular numerals.
Wordmark: "SWARM" (Inter 300, letter-spacing 0.18em) + "GUARD"
(Bricolage 800), preceded by a small line-art drone mark.

=== 5.3 SHAPES, SPACING, SHADOWS ===
Outlines: 1.5px ink on cards; 2px ink on buttons and inputs.
Radius: cards 16px, buttons 10px, inputs 10px, pills 999px.
Hard shadows (no blur): buttons 3px 3px 0 ink; feature cards 4px 4px 0 ink;
dense data cards no shadow, only the outline (keeps dashboards calm).
Console panels: radius 16px, 1px console_ln border, no shadow.
Spacing: 8px grid; 24px gutters; max content width 1360px; generous
vertical rhythm (48px between page sections).

=== 5.4 MOTION SYSTEM (assets/style.css) ===
Easing: cubic-bezier(0.2, 0.8, 0.2, 1). Durations: 90ms press, 140ms hover,
240ms entrances, 400ms count-up/draw.
Buttons (.stButton > button, .stDownloadButton > button, .sg-btn):
  rest   lavender fill, ink text, 2px ink border, 3px 3px 0 ink shadow
  hover  translate(-1px,-1px), shadow 4px 4px 0 ink, fill lavender_2
  active translate(3px,3px), shadow 0 0 0 ink (presses into its shadow)
  ripple a CSS-only ::after circle in lavender_2 that scales from the
         centre and fades on :active
  focus-visible 2px forest outline, offset 3px
  secondary variant: paper fill; destructive variant (tamper test only):
  red fill, white text, same outline and shadow.
Page entrance: main blocks fade in + translateY(8px -> 0) in 240ms, staggered
  40ms per block via nth-child delays (max 6 steps).
KPI count-up: CSS-only using @property --num (integer) and a counter,
  animating 0 -> value in 400ms; falls back to a static number in browsers
  without @property.
Tabs: the selected-tab indicator slides between tabs (style Streamlit's tab
  highlight element with a 240ms transform transition), forest colour.
Toggles/checkboxes/sliders: forest accent, smooth 140ms state change.
Verification PASS: an SVG checkmark draws itself (stroke-dashoffset, 400ms).
Verification FAIL: the failing row shakes horizontally once (240ms) and
  turns red-tinted.
Toasts (st.toast) restyled: paper card, ink outline, small hard shadow.
@media (prefers-reduced-motion: reduce): disable all transforms and
  animations; keep instant state changes.
Honest limitation (record in notes): Streamlit reruns the script on
interaction, so there are no animated transitions BETWEEN pages; the
per-page entrance animations provide the smooth feel.

=== 5.5 GRAPHICS (assets/stickers.py) ===
Inline SVG line-art, 2px ink outlines, fills paper / lavender / blush,
optional tilt between -4 and 4 degrees for a hand-placed feel:
  drone (quadcopter, top view), drone (side view), satellite, radar
  sweep, signal waves, GNSS constellation dots, shield with check, padlock,
  fingerprint (forensics), document with seal (evidence), crosshair.
Use them only on Home, scenario cards, empty states and section headers,
never inside charts. Helper: sticker(name, size=72, rotate=0) -> SVG.

=== 5.6 HOME HERO (ui/hero.py rendering assets/hero.html) ===
Render with streamlit.components.v1.html (an iframe), so SVG animation and
fonts work reliably; transparent background; fixed height ~620px desktop,
~760px mobile; fonts loaded inside the iframe.
Layout (inspired by Wispr Flow's hero, rebuilt for drones):
  - eyebrow: "SWARMGUARD · GPS INTEGRITY FOR DRONE SWARMS"
  - headline (Instrument Serif): "Detect the drift," / italic "prove the attack."
  - two-line subtitle (Inter): "Simulates GPS spoofing on a drone swarm,
    catches it with three independent detectors, and seals the evidence."
  - lavender tactile CTA "Run a scenario" + text link "How it works"
    (the iframe cannot call Streamlit directly; place the real
    st.button CTAs directly under the iframe, styled identically, and keep
    the in-iframe button as the visual focal point linking to an anchor)
  - background art: a soft curved flight path (SVG path, ink 1px at 25%
    opacity) with live telemetry text flowing along it using <textPath>
    and an animated startOffset ("GPS 18.5204N 73.8567E · NIS 2.1 · range
    7.98 m · ..."), and a small line-art drone gliding along the path with
    <animateMotion>
  - a waveform pill (thick ink outline, as in Wispr's mic pill) whose bars
    animate like an NIS signal; one bar periodically spikes and turns red
    for a moment (a detection)
  - a black ribbon band crossing the bottom at a slight angle, scrolling
    detection events in white Inter text ("Drone 04 · motion check ·
    ALARM · t = 20.3 s ··· swarm pattern · A silent, B and C alarm ···")
  - all motion slow and calm; pauses under prefers-reduced-motion

=== 5.7 .streamlit/config.toml ===
base light; backgroundColor ivory; secondaryBackgroundColor paper; textColor
ink; primaryColor forest; font sans serif; client.toolbarMode "minimal";
hide default sidebar page navigation.

=== 5.8 assets/style.css ===
- Google Fonts import for the four families, with fallbacks.
- Hide Streamlit header decoration, main menu, footer and deploy button.
- Global: ivory background, ink text, max width 1360px, 24px gutters.
- Navbar (.sg-nav): a floating pill like Wispr's: paper background,
  1.5px ink outline, radius 999px, centred, sticky at the top with 12px
  margin; wordmark left, page links centre (active link forest with a
  sliding underline), a small status pill and the current scenario right.
  Use st.navigation(position="top") if the installed Streamlit supports
  it, styled to match; otherwise keep the custom navbar and a styled
  sidebar.
- Classes: .sg-card, .sg-card--feature (hard shadow), .sg-console,
  .sg-kpi, .sg-pill-{status}, .sg-eyebrow, .sg-section, .sg-feed,
  .sg-ribbon, .sg-inspector, .sg-mono, .sg-enter (entrance animation).
- Restyle Streamlit widgets: inputs, selects, number inputs, sliders,
  expanders, tabs, radio, checkbox, toggle, dataframes, st.code, alerts.
  All get 2px or 1.5px ink outlines, 10px radius, Inter, no default grey.
- Alerts (st.info/success/warning/error) restyled as outlined cards with a
  coloured left bar and a text label; never stacked more than one.

=== 5.9 ui/components.py ===
  navbar(active_page)
  eyebrow(text); section(title, caption=None)
  kpi_card(label, value, status=None, delta=None) and kpi_row(items):
    Bricolage numerals with count-up; a thin status bar on the bottom edge
  status_pill(status): outline + tinted fill + uppercase text label
  console_panel(title, body_fn): dark panel wrapper for map/feed/verify
  drone_table(step_results): st.dataframe with column_config (progress
    bars for scores, formatted numbers, status as text, no index)
  event_feed(alerts): inside a console panel, mono timestamps, status dots,
    newest first, smooth entrance per row
  ribbon(events): black scrolling band (used on the Dashboard too, slower)
  drone_inspector(drone, data): outlined "properties" card: per-detector
    status and numbers, fusion verdict, explainer reasons, recommended action
  scenario_card(scenario): feature card with a sticker, title, one-line
    description, expected outcomes as small pills, hover lift, click to run
  verification_table(report): PASS rows draw a check; FAIL rows shake once
  empty_state(message, action_label): sticker + one sentence + one button
  explain_box(text): "How to read this" expander

=== 5.10 ui/plots.py ===
Two Plotly templates:
  "swarmguard_light" (charts on paper cards): transparent background, Inter,
    ink axis lines 1px, very light dashed gridlines (line colour), muted tick
    labels, data lines 2.25px, threshold lines dashed ink with
    right-aligned labels, legend horizontal above, no in-figure titles,
    smooth transitions (layout.transition duration 300ms).
  "swarmguard_console" (charts inside console panels): console background,
    console_ln gridlines, console_tx labels, mint as the neutral trace.
swarm_map(...): lives inside a console panel, always the dark basemap
  "carto-darkmatter" over Pune (centre 18.5204, 73.8567), coordinates via
  enu_to_latlon. Drones: circles with a 1px ivory outline, fill = status
  colour; trails in mint at decreasing opacity (last 20 positions);
  anchored drones as diamonds; formation links as thin mint lines at 20%
  opacity (toggle). Frames every 5 steps with play/pause and a styled time
  slider (WHY: 600 frames lag the browser). Light-show mode adds a larger
  low-opacity marker layer under each drone so formations read like lights
  at night. Evaluation view toggle: true positions as hollow markers joined
  to GPS by dashed lines, with a badge "Evaluation view: ground truth".
  Offline map toggle: same data on a plain dark metric grid.
Other figures (light template unless inside a console): detector_timeseries
  (A vote fraction 0.5 line; NIS log axis with threshold; CUSUM with h; C
  residual with tau_c; onset line labelled "simulation onset"),
  residual_heatmap (paper -> blush -> red scale), comparison_bar with Wilson
  error bars, detection_curve with CI band, delay_box,
  stealth_envelope_chart, propagation_heatmap, pagerank_bar, recovery_map,
  fusion_chart (belief + plausibility), min_separation_chart.
  All use detector and status colours from the palette.

=== 5.11 ui/state.py ===
Session keys: scenario_key, sim_cfg, atk_cfg, seed, sim_result,
detection_runs, fusion, recovery, incident_id. get_or_run_pipeline():
simulate -> detect -> fuse -> recover, cached with st.cache_data keyed on
config hashes + seed + detector config hash; uses the saved calibration for
the scenario's swarm configuration (as built in Phase 2); opens and stores
an incident if any detector alarms.

=== 5.12 app.py ===
st.set_page_config(page_title="SwarmGuard", layout="wide",
page_icon=":material/radar:"); inject style.css; navbar; st.navigation with
sections and Material icons:
  Operations: Home, Dashboard, Swarm Simulator, Attack Lab
  Analysis: Detection Analysis, Experiments, Recovery
  Forensics: Incidents, Evidence Centre
  Reference: Methodology
Each st.Page points to a file in views/.

=== 5.13 VIEWS ===
home.py: animated hero (5.6); real CTA buttons; three feature cards
  (Detect / Recover / Prove), each with a sticker and two lines of copy;
  a section "Why one detector isn't enough" with the cancellation equation
  shown in a clean card (st.latex) and one sentence of explanation; the
  architecture as an st.graphviz_chart styled in palette colours inside a
  paper card; four scenario cards (single_hijack, slow_drift,
  rigid_swarm_shift, light_show_pune) that run the pipeline and
  st.switch_page to the Dashboard.
dashboard.py: eyebrow + serif title; KPI row (total, normal, suspicious,
  compromised, no evidence); swarm map console panel with step slider;
  drone table; event feed console panel; slow ribbon of recent events;
  latest incidents; a pattern banner (outlined, red bar) when fusion detects
  one, e.g. "Coordinated swarm shift suspected".
simulator.py: scenario picker + seed; animated map; Evaluation view and
  Offline map toggles; min-separation chart; legend card for slots,
  anchors and links.
attack_lab.py: left column presets or manual controls in outlined expanders
  (Swarm, Sensors, Anchors, Manoeuvres, Attack, Detector thresholds with
  "use calibrated"); right column one primary RUN button, results table
  (per detector: detected, delay, identified drones, false alarms),
  expected-vs-actual check, link to the auto-created incident. Friendly
  validation messages, never tracebacks.
detection.py: drone selector; detector timeseries; residual heatmap with
  step slider; comparison table; fusion chart; agreement matrix (drones ×
  detectors, labelled); Drone Inspector in a right column; explain_box under
  each chart.
experiments.py: tabs Calibration / Sweeps / Evasion / Propagation
  (propagation shows both centre_gain values per DESIGN DECISION D1,
  labelled); runtime warning before long jobs; styled progress bars; cached
  results.
recovery.py: recovery map with inlier/outlier anchors; estimated attack
  vector; multilateration results per flagged drone, including
  INSUFFICIENT_NEIGHBOURS; before/after error table labelled as evaluation.
incidents.py: filterable table; select -> timeline figure, explainer text,
  detector details; side-by-side comparison of two incidents.
evidence.py: pick an incident -> build package if missing; download buttons;
  VERIFY EVIDENCE (primary) -> verification table in a console panel with
  draw-on checks; Tamper Test (destructive style) on a copy with
  before/after side by side and the changed byte highlighted; custody
  chain table; public key fingerprint and PEM in an expander; a callout on
  what a signature does and does not prove.
methodology.py: serif title; st.latex for every equation (pairwise residual,
  majority vote, Kalman predict/update, NIS, CUSUM, anchor residual + median
  example, Dempster's rule, MDS double centering, Kabsch, multilateration
  residual, Wilson interval); assumptions; blind-spot table; fixes F1–F8;
  design decisions D1–D4 (and any later ones in CLAUDE.md); credits R6
  (also credit Wispr Flow and Urban Macaw as visual design references);
  limitations.

EMPTY AND ERROR STATES
Every view works on a fresh database: empty_state(), never a blank page or
a traceback.

RESPONSIVE
Check at 1440px, 1024px and 390px widths. Columns stack on narrow screens;
the hero scales down; tables scroll horizontally inside their card.

QUALITY BAR (check every view before finishing)
- Same outline weights, radii, shadows and spacing everywhere.
- No element still shows Streamlit's or Plotly's default look.
- Every status shows text, not just colour.
- Text contrast meets WCAG AA on ivory and on console panels.
- Animations feel quick and calm; nothing blocks interaction.
- Take screenshots of every view and save them to docs/screenshots/
  (needed for the presentation).

ACCEPTANCE CHECKS
  .venv\Scripts\streamlit run app.py
  every view loads on a fresh database with no exceptions
  buttons lift on hover, press into their shadow on click, and ripple
  the Home hero animates (drone on path, flowing telemetry, waveform,
  ribbon) and stops under reduced motion
  the light_show_pune quick start runs end to end on the dark map and
    produces an incident whose evidence package verifies with the
    draw-on checkmarks; the tamper test shows the shake and FAIL
  the regenerated PDF uses the new palette
############################################################################