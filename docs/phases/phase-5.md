############################################################################
# PHASE 5 — STREAMLIT APPLICATION, NEO-BRUTALIST "POSTER" DESIGN
# Files: .streamlit/config.toml, assets/style.css, assets/stickers.py,
#        ui/theme.py, ui/state.py, ui/components.py, ui/plots.py, app.py,
#        views/*.py
# This file OVERRIDES the "UI DIRECTION" summary in CLAUDE.md. Before
# writing code, update the UI DIRECTION section of CLAUDE.md to describe
# this design (cream + black + red neo-brutalist poster style).
############################################################################

DESIGN DIRECTION
Neo-brutalist retro poster style, inspired by a reference e-commerce site
(urbanmacaw.com, screenshot in docs/design-refs/ if present). Warm cream
background, thick black outlines, hard offset shadows with no blur, bold
quirky headings, a deep red hero band with grainy texture, giant blush-pink
serif lettering behind illustrations, sticker-style line art, and a
checkerboard band. Confident and handmade, never generic. The project is a
cybersecurity tool, so: the Home page is the playful poster; analysis pages
use the same visual language but calmer (no stickers inside chart areas,
generous white space, charts readable first).

BANNED
- emojis anywhere (use our SVG stickers or Streamlit Material icons)
- gradients, glows, glassmorphism, blurred shadows
- default st.metric boxes for main KPIs; default Streamlit look anywhere
- Plotly default theme/colours
- cartoon faces (eyes, smileys); stickers must be drones, radar, satellites,
  signal waves, shields, padlocks, bolts, stars, sparkles, clouds
- more than the palette below

=== 5.1 ui/theme.py (single source of truth) ===
Palette:
  bg        #F4F1EA  warm cream (page)
  surface   #FFFFFF  cards, buttons
  ink       #111111  text, borders, shadows
  ink_muted #5C5A55  secondary text
  red       #C8221B  brand red (hero bands, primary emphasis)
  red_deep  #9E1A14  hover/pressed red, texture shade
  blush     #F4B6B2  giant background lettering, soft fills
  checker   #111111 / #F4B6B2 for checkerboard bands
Status colours (always paired with a text label):
  NORMAL #1F9D55, SUSPICIOUS #E8A100, ALARM/COMPROMISED #C8221B,
  NO_EVIDENCE/INSUFFICIENT #8A8577
Detector colours, fixed everywhere: A #2E5BFF, B #12A37F, C #E8A100, D #7A3FF2
Fonts (Google Fonts, with system fallbacks):
  display:  "Bricolage Grotesque" 800, for page titles and big headings
  poster:   "DM Serif Display", ONLY for giant background lettering on Home
  body:     "Inter" 400/500/600, for UI text
  mono:     "JetBrains Mono", for numbers, coordinates, hashes, IDs
Scale: page title 36px display; section heading 22px display; body 15px;
  data 13px mono with tabular numerals.
Shapes: border 2.5px solid ink; radius 14px (cards), 12px (buttons),
  999px (pills); hard shadow 5px 5px 0 ink (cards), 4px 4px 0 ink (buttons).
Spacing: 8px grid; 24px gutters; max content width ~1400px.
Also exports the Plotly template and status -> colour/label maps.

=== 5.2 .streamlit/config.toml and assets/style.css ===
config.toml: theme base light, backgroundColor bg, secondaryBackgroundColor
surface, textColor ink, primaryColor red; client.toolbarMode = "minimal";
hide default sidebar page navigation (we use st.navigation).
style.css (injected once by app.py):
- Google Fonts import (Bricolage Grotesque, DM Serif Display, Inter,
  JetBrains Mono) with fallbacks.
- Hide Streamlit header decoration, main menu, footer and deploy button.
- Top navbar (.sg-nav): cream background, 2.5px bottom border in ink,
  wordmark left "SWARM" (Inter 300, letter-spaced) + "GUARD" (Bricolage
  800 italic), mirroring the reference site's two-weight logo.
- BUTTONS (.stButton > button, .stDownloadButton > button, .sg-btn):
    background surface, color ink, border 2.5px ink, radius 12px,
    font Bricolage Grotesque 700 uppercase, letter-spacing 0.02em,
    box-shadow 4px 4px 0 ink, transition transform 80ms, box-shadow 80ms.
    :hover  -> transform translate(-2px,-2px); box-shadow 6px 6px 0 ink
    :active -> transform translate(4px,4px);  box-shadow 0 0 0 ink
               (the button "presses into" its shadow)
    :focus-visible -> 3px outline in red, offset 3px
    Primary variant (kind="primary"): background red, color white, same
    border/shadow; hover background red_deep. Max one primary per screen.
- Cards (.sg-card): surface background, 2.5px ink border, radius 14px,
  shadow 5px 5px 0 ink; hoverable cards lift like buttons.
- KPI (.sg-kpi): card with a muted uppercase label, a 40px Bricolage
  value, an optional coloured status bar along the bottom edge.
- Pills (.sg-pill-{status}): 2px ink border, status-colour fill at about 25%
  tint, ink text, uppercase 12px.
- Inputs, selects, sliders, expanders, tabs, dataframes: 2px ink borders,
  12px radius, no default grey look; selected tab gets a red underline
  block.
- Hero (.sg-hero): full-width deep-red band, 2.5px ink top/bottom borders,
  grainy texture via an inline SVG feTurbulence noise data-URI at about 18%
  opacity, giant blush DM Serif Display word behind the content.
- Checkerboard (.sg-checker): CSS repeating-conic-gradient, ink/blush squares
  of 28px, used as a curved band in the hero corner and as a thin divider.
- Decorative swirl lines: thin cream/blush SVG curves in hero corners.
- Light, 120ms entrance animation for stickers (scale 0.9 -> 1); respect
  prefers-reduced-motion.

=== 5.3 assets/stickers.py ===
Inline SVG strings (no external images), all with 3px ink outlines, white or
blush fill, slightly tilted (-8deg to 8deg) for a hand-placed feel:
  drone (quadcopter top view), satellite, radar dish, signal waves,
  shield, padlock, lightning bolt pair, 4-point sparkle, 5-point star,
  cloud, crosshair. Helper sticker(name, size, rotate) -> SVG string.

=== 5.4 ui/components.py (st.html / st.markdown + CSS classes) ===
  navbar(active_page): wordmark + page links + small status pill on the right
  hero(title_bg, headline, tagline, cta_label, stickers): Home poster band
  section(title, caption=None): Bricolage heading + muted caption
  kpi_card(label, value, status=None) and kpi_row(items)
  status_pill(status): always text + colour, never colour alone
  drone_table(step_results): st.dataframe with column_config (progress bars
    for scores, formatted numbers, no index)
  event_feed(alerts): bordered timeline list, mono timestamps, status dots
  drone_inspector(drone, data): bordered "properties" card: per-detector
    status and numbers, fusion verdict, explainer reasons, recommended action
  scenario_card(scenario): clickable bordered card with a sticker, title,
    one-line description, expected outcomes as small pills
  verification_table(report): PASS/FAIL rows with icons and text
  empty_state(message, action_label): sticker + one sentence + one button
  explain_box(text): "How to read this" expander

=== 5.5 ui/plots.py (Plotly template "swarmguard") ===
Template: white plot background inside a bordered card, cream paper
(transparent), ink axis lines 1.5px, very light dashed gridlines, Inter
font, ink tick labels, no in-figure titles, data lines 2.5px, threshold
lines dashed ink with right-aligned labels, legend horizontal above.
Figures:
  swarm_map(...): real map of Pune via Plotly map traces, centred on
    (18.5204, 73.8567), coordinates from enu_to_latlon.
    Default style "carto-positron" (light, matches cream).
    Light-show mode uses "carto-darkmatter" (a night show), with drones as
    bright markers plus a larger low-opacity glow layer.
    Drones as circles with ink outline, fill = status colour; faint trails
    (last 20 positions); anchored drones as diamonds; formation links as
    thin lines (toggle). Animation frames every 5 steps with play/pause and
    a time slider (WHY: 600 frames lag the browser).
    Evaluation view toggle: true positions as hollow markers joined to GPS
    by dashed lines, with a red-bordered "Evaluation view: ground truth"
    badge.
    Offline map toggle: same data on a plain metric grid (for demos
    without internet).
    The whole map sits inside a .sg-card.
  detector_timeseries, residual_heatmap (white -> blush -> red scale),
  comparison_bar (detector colours, Wilson error bars), detection_curve,
  delay_box, stealth_envelope_chart, propagation_heatmap, pagerank_bar,
  recovery_map, fusion_chart, min_separation_chart. Same specs as before;
  all use the template and detector colours.

=== 5.6 ui/state.py ===
Session keys: scenario_key, sim_cfg, atk_cfg, seed, sim_result,
detection_runs, fusion, recovery, incident_id. get_or_run_pipeline():
simulate -> detect -> fuse -> recover, cached with st.cache_data keyed on
config hashes + seed + detector config hash; opens and stores an incident if
any detector alarms.

=== 5.7 app.py (router) ===
st.set_page_config(page_title="SwarmGuard", layout="wide",
page_icon=":material/radar:"); inject style.css; render navbar;
st.navigation with sections and Material icons:
  Operations: Home, Dashboard, Swarm Simulator, Attack Lab
  Analysis: Detection Analysis, Experiments, Recovery
  Forensics: Incidents, Evidence Centre
  Reference: Methodology
Each st.Page points to a file in views/.

=== 5.8 views ===
home.py, the poster page:
  Hero band: deep red textured background, giant blush "SWARMGUARD" in DM
  Serif Display behind a sticker composition (drone swarm in formation,
  satellite, radar, signal waves, bolts, sparkles, clouds), a checkerboard
  curve in the bottom-left corner, swirl lines, tagline in white
  ("Detect. Recover. Prove."), and a white neo-brutalist CTA button
  "RUN A SCENARIO".
  Below the hero, a huge Bricolage headline (e.g. "GPS spoofing defence for
  drone swarms."), three bordered value cards (Detect / Recover / Prove)
  each with a sticker, the architecture as a clean st.graphviz_chart in a
  bordered card, and four scenario cards (single_hijack, slow_drift,
  rigid_swarm_shift, light_show_pune) that run the pipeline and
  st.switch_page to the Dashboard.
dashboard.py: navbar status; KPI row (total, normal, suspicious,
  compromised, no evidence); swarm map with step slider; drone table;
  event feed; latest incidents; a red bordered banner when fusion detects a
  pattern ("Coordinated swarm shift suspected").
simulator.py: scenario picker + seed; animated map; Evaluation view and
  Offline map toggles; min-separation chart; legend card for slots,
  anchors, links.
attack_lab.py: left column presets or manual controls in bordered expanders
  (Swarm, Sensors, Anchors, Manoeuvres, Attack, Detector thresholds with
  "use calibrated"); right column one primary RUN button, results table
  (per detector: detected, delay, identified drones, false alarms),
  expected-vs-actual check, link to the auto-created incident. Friendly
  validation messages, never tracebacks.
detection.py: drone selector; detector timeseries; residual heatmap with
  step slider; comparison table; fusion chart; agreement matrix (drones ×
  detectors, labelled); Drone Inspector in a right column; explain_box under
  each chart.
experiments.py: tabs Calibration / Sweeps / Evasion / Propagation (both
  centre_gain values per DESIGN DECISION D1, labelled); runtime warning
  before long jobs; progress bars; cached results.
recovery.py: recovery map with inlier/outlier anchors; estimated attack
  vector; multilateration results per flagged drone, including
  INSUFFICIENT_NEIGHBOURS; before/after error table labelled as evaluation.
incidents.py: filterable table; select -> timeline figure, explainer text,
  detector details; side-by-side compare of two incidents.
evidence.py: pick an incident -> build the package if missing; download
  buttons; VERIFY EVIDENCE (primary) -> verification table; Tamper Test on
  a copy with before/after side by side and the changed byte; custody chain
  table; public key fingerprint and PEM in an expander; callout on what a
  signature does and does not prove.
methodology.py: st.latex for every equation (pairwise residual, majority
  vote, Kalman predict/update, NIS, CUSUM, anchor residual + median example,
  Dempster's rule, MDS double centering, Kabsch, multilateration residual,
  Wilson interval); assumptions; blind-spot table; fixes F1–F8; design
  decisions D1–D4; credits R6; limitations.

EMPTY AND ERROR STATES
Every view works on a fresh database with empty_state(), never a blank page
or traceback.

QUALITY BAR
Check every view at about 1440px and 1024px width. Borders, shadows and
spacing must be consistent on every element. Any element still showing
Streamlit's default look must be restyled.

ACCEPTANCE CHECKS
  .venv\Scripts\streamlit run app.py
  every view loads on a fresh database with no exceptions
  buttons visibly lift on hover and press into their shadow on click
  the light_show_pune quick start runs end to end on the night map and
    produces an incident whose evidence package verifies
############################################################################