# Phase 5 Explained: The Streamlit App ("Ivory + Console")

Phases 1 to 4 built the science: a simulator, three detectors, fusion,
recovery, statistics and signed evidence. All of it ran from the command
line. Phase 5 puts it in a web app that an examiner, a teammate or a
presentation audience can use without typing a command. It answers three
questions:

1. **How does someone run and explore an attack?** Ten pages ("views")
   behind a top navigation bar.
2. **How do all pages talk about the same flight?** Shared session state
   and a cache (`ui/state.py`).
3. **How does it look professional and stay honest?** One design system
   (`core/palette.py`, `ui/theme.py`, `assets/style.css`), with status always
   shown as text, "simulation result" labels, and an explicit
   "Evaluation view" whenever ground truth is shown.

```
streamlit run app.py
   │
   ├─ inject_css()        style.css, with colours generated from core/palette.py
   ├─ st.navigation(top)  Operations | Analysis | Forensics | Reference
   ├─ navbar()            wordmark, page name, run status pill, scenario tag
   └─ view.py             one of the ten pages
            │
            ▼
   ui/state.run_scenario() / run_configs()
            │  run key = sha256(sim + attack + detector configs)
            ▼
   _cached_pipeline()     st.cache_data: simulate → detect → recover → fuse
            │
            ▼  (outside the cache: it writes to the database)
   open_incident()        reuses the incident if this run key was seen before
```

---

## 1. `.streamlit/config.toml`: Streamlit's own theme

Streamlit draws its widgets (sliders, inputs, tables) with its own theme.
This file sets that theme to our colours: ivory page, white "paper"
secondary background, ink text, forest as the primary colour, 10 px radii.
It also hides Streamlit's automatic sidebar page list (we build our own
navigation) and the developer toolbar.

**Why a test checks it:** the colours in this file are typed in by hand
(TOML cannot import Python). `test_config_toml_matches_palette` compares
them with `core/palette.py`, so the two can never drift apart.

---

## 2. `ui/theme.py`: the design tokens

A "design token" is a named design value: a colour, a radius, a duration.
This module holds every non-colour token and re-exports the colours:

| Token | Value | Used for |
|---|---|---|
| card radius | 16 px | cards, console panels |
| button / input radius | 10 px | buttons, inputs |
| outlines | 1.5 px (cards), 2 px (buttons) | ink borders |
| hard shadows | 3 px buttons, 4 px feature cards | offset, **no blur** |
| durations | 90 ms press, 140 ms hover, 240 ms entrance, 400 ms count-up/draw | motion |
| fonts | Instrument Serif, Bricolage Grotesque, Inter, JetBrains Mono | titles, numbers, body, data |

`css_variables()` turns the palette and tokens into CSS custom properties,
for example `--sg-forest: #0B4D40;` and `--sg-dur-hover: 140ms;`.
`stylesheet()` puts those variables in front of `assets/style.css`, and
`inject_css()` adds the result to every page.

**Why generate the CSS variables?** So `style.css` contains **no colour
values at all**: it only says `var(--sg-forest)`. A test fails if anyone
types a hex colour into `style.css` or `hero.html`. Changing a colour in
`core/palette.py` therefore changes the app, the hero **and** the PDF.

Helpers: `status_label("insufficient_evidence")` → "No evidence";
`status_tone("compromised")` → `"alarm"` (which CSS class to use);
`rgba("#0B4D40", 0.2)` → `"rgba(11,77,64,0.2)"` for Plotly bands.

---

## 3. `assets/style.css`: how everything looks and moves

Organised in numbered sections: base, navigation, typography, entrance
motion, tactile buttons, widgets, alerts/tabs/expanders, cards and console
panels, KPIs and pills, feed and ribbon, verification motion, responsive
rules and reduced motion. Key ideas:

- **Hooks.** Streamlit marks its elements with `data-testid` attributes
  (`stTopNav`, `stPlotlyChart`, ...). The CSS styles those. Containers we
  create with `st.container(key="sg-console-map")` get the class
  `st-key-sg-console-map`, so one rule `[class*="st-key-sg-console"]`
  paints every console panel.
- **Tactile buttons.** At rest a button sits on a hard 3 px ink shadow. On
  hover it moves up-left 1 px and the shadow grows to 4 px (it "lifts"). On
  press it moves 3 px down-right and the shadow becomes 0 (it "presses into
  its shadow"). The ripple is a `::after` circle scaled from 0 to 1 while
  pressed: CSS only, no JavaScript.
- **KPI count-up.** The number is drawn by a CSS counter whose value is an
  animatable `@property` integer, so it counts from 0 to the target in
  400 ms. A hidden copy of the number is kept for screen readers.
- **Draw-on check.** A PASS check icon is revealed left to right with
  `clip-path: inset(0 100% 0 0)` → `inset(0 0 0 0)`, rows staggered by 60 ms.
  A FAIL row turns red-tinted and shakes once (`translateX` ±6 px).
- **Reduced motion.** Under `prefers-reduced-motion: reduce` every animation
  and transition duration becomes 0 and hover lifts are switched off.
- **Responsive.** Below 900 px the padding and type shrink; below 640 px
  columns stack and KPIs go two per row. On phones Streamlit turns the top
  navigation into a sidebar, which is styled to match.

---

## 4. `assets/stickers.py`: line-art graphics

Eleven technical line drawings (drone from above and from the side,
satellite, radar, signal, constellation, shield, padlock, fingerprint,
document, crosshair), built as SVG strings: thick ink outlines with paper,
lavender or blush fills. `sticker(name, size, rotate)` returns the SVG.
Rotation is limited to ±4° so stickers look placed by hand, not tilted.

`svg_img()` / `sticker_img()` wrap an SVG as
`<img src="data:image/svg+xml;base64,...">`. **Why:** `st.html` cleans
(sanitises) the HTML it is given and removes every `<svg>` element, so an
inline drawing simply disappeared. An image with a data URI survives the
cleaning and looks identical. (Found in the browser check, see section 13.)

---

## 5. `assets/hero.html` + `ui/hero.py`: the animated Home banner

The hero is a small HTML page with placeholders (`__HEADLINE_1__`,
`/*SG_VARS*/`, ...). `hero_html()` fills them, escaping every text with
`html.escape`, and inserts the generated colour variables. `render_hero()`
shows it in an iframe (`st.iframe`).

What moves (all slow and calm, and the only looping motion in the app):
- telemetry text flowing along a curved flight path (SVG `<textPath>` with
  an animated `startOffset`);
- a small drone gliding along the same path (`<animateMotion>`);
- a waveform pill whose bars pulse like an NIS signal, one bar
  periodically spiking red (a detection);
- a black ribbon of detection events crossing the bottom at an angle.

**Why an iframe?** `st.html` would strip the SVG animation and the script.
Only our own template is ever loaded, never user input.
**Why real buttons underneath?** An iframe cannot press Streamlit buttons.
The hero's "Run a scenario" link scrolls the page to the real buttons.
**Reduced motion:** a tiny script calls `svg.pauseAnimations()` and CSS
stops the bars and ribbon. Verified in a browser: `animationsPaused()` is
`true` and all animation durations are 0 under reduced motion.

---

## 6. `ui/components.py`: reusable pieces

Every component has two halves:
- a **pure builder** (`kpi_card_html(...)`, `verification_table_html(...)`)
  that returns an HTML string and is unit-tested without Streamlit;
- a thin **wrapper** (`kpi_card(...)`) that puts it on the page.

All text goes through `html.escape`, so a file name like `<script>` is shown
as text, never run. Components: wordmark and context bar, page header and
section headings, KPI cards, status and detector pills (always with a text
label), console panels, event feed, ribbon, pattern banner, drone table
(`st.dataframe` with progress-bar columns), drone inspector, scenario
cards, verification table, empty state, "How to read this" box, paper card,
red tamper button, and `require_run()`.

`require_run()` is used by every analysis page: it returns the current run,
or shows an empty state ("No simulation has run yet") with one button that
runs the quick-start scenario. **Why:** every page must work on a fresh
database, with no blank page and no traceback.

`chart(fig)` calls `st.plotly_chart(..., theme=None)`. **Why `theme=None`:**
otherwise Streamlit re-themes Plotly charts and overrides our templates.

---

## 7. `ui/plots.py`: every figure

**Two Plotly templates.** `swarmguard_light` for charts on paper cards
(transparent background, ink axes, dashed light gridlines, 2.25 px lines)
and `swarmguard_console` for charts inside console panels (dark background,
mint as the neutral colour). Threshold lines are dashed ink with a label;
the attack onset is a dotted line labelled "simulation onset", since only
the simulation knows it.

**The swarm map** (`swarm_map`). Positions are metres (x east, y north) and
are converted for display only (rule R3):

```
lat = 18.5204 + y / 111320
lon = 73.8567 + x / (111320 · cos(18.5204°))
```

The basemap is always the dark `carto-darkmatter` style inside a console
panel. Drones are coloured by fused verdict, with a mint trail of their last
20 positions. Animation frames are taken **every 5th step**: 600 steps
would give 600 frames and make the browser lag; 120 frames still look
smooth. Options: formation links, Evaluation view (true positions as hollow
markers joined to GPS by lines, labelled as ground truth), offline grid
(no internet needed), light-show glow.

**Fitting the map** (`_zoom_for`). Web maps halve the metres per pixel for
each zoom level. Plotly's map uses MapLibre, whose tiles are 512 px wide:

```
metres per pixel = 78271.52 · cos(latitude) / 2^zoom
zoom = log2( 78271.52 · cos(latitude) / (1.3 · extent / pixels) )
```

The 1.3 adds a 30 % margin; `extent` is the larger side of the flight's
bounding box and the camera centres on the box's middle. (The first version
used the 256 px constant 156543, which zoomed in twice too far; caught in
the browser check and now tested.)

Other figures: detector time series (A vote share, B NIS on a log axis,
B CUSUM, C residual, each with its threshold), Detector A's pair-residual
heatmap (paper → blush → red), belief/plausibility, alarm timeline,
detection-rate curves with Wilson bands, delay boxes, stealth envelope,
calibration curve, propagation heatmap, PageRank bars, minimum separation,
recovery map, attack-vector series, and the architecture diagram as
Graphviz DOT in palette colours.

---

## 8. `ui/state.py`: one current run, shared by all pages

**Session state.** `st.session_state` is a per-browser-tab dictionary that
survives page changes. It holds the current run (`pipeline`, `sim_cfg`,
`atk_cfg`, `det_cfg`, `seed`, `scenario_key`, `incident_id`, ...), so the
Dashboard, Detection, Recovery and Evidence pages all describe the same
flight.

**Caching.** Streamlit re-runs the page script on every click. Re-simulating
each time would be slow, so `_cached_pipeline` is wrapped in
`st.cache_data`, keyed on the **run key**:

```
run_key = sha256( canonical_json(sim_config) + canonical_json(attack_config)
                  + canonical_json(detector_config) )
```

The sim config includes the seed. Same inputs give identical output (rule
R2), so a stored result can safely be reused.

**Side effects stay outside the cache.** Opening an incident writes to the
database, so it must not be cached (a cache would skip it). It runs after
the cached call; `open_incident()` looks up the run key first, so revisiting
a run never creates a second incident.

**Thresholds.** If no detector config is given, `load_or_default(sim)` uses
the saved calibration for that swarm configuration, else the analytical
defaults, and logs which (fix F3).

**Pure helpers** (unit-tested): `build_events` (first alarm per detector and
drone, newest first, so one long alarm is one event, not 400 rows),
`verdict_counts`, `drone_rows`, `inspector_data` (with the recommended
action, which never claims "genuine"), `agreement_matrix` (text cells:
ALARM / silent / no evidence), `comparison_table`, `detection_results_table`
and `expected_vs_actual` (labelled as evaluation, since they use the attack
labels), `pattern_banner_text`, `affected_text` ("3,6" → "Drone 04, Drone 07").

**Paths** come from environment variables (`SWARMGUARD_DB`,
`SWARMGUARD_KEYS`, `SWARMGUARD_PACKAGES`), which lets the tests start every
page on an empty temporary database.

---

## 9. `app.py`: the router

Sets the page title and icon, injects the stylesheet, builds
`st.navigation(..., position="top")` with four sections and Material icons,
draws the context bar, and runs the chosen page.

**Why `views/` and not `pages/`?** Streamlit automatically builds a menu
from a folder called `pages/`, which would conflict with ours.

---

## 10. The ten views

| View | What it shows | Notable decisions |
|---|---|---|
| **Home** | hero, start buttons, Detect/Recover/Prove cards, the cancellation equation, architecture diagram, 4 scenario cards | each scenario card runs the pipeline and opens the Dashboard (`st.switch_page`) |
| **Dashboard** | KPIs, pattern banner, map, event feed, ribbon, drone table, latest incidents | **one time slider drives every panel**, so they always agree |
| **Swarm Simulator** | scenario + seed, animated map, Evaluation / Offline / Links toggles, closest-approach chart | min separation uses truth, so it is labelled evaluation |
| **Attack Lab** | preset, then Swarm / Sensors / Anchors / Manoeuvres / Attack / Thresholds; Run; results; expected vs actual | widget keys include the preset name, so switching preset resets the controls; invalid settings show "These settings cannot be simulated: …" (the config's own `ValueError`), never a traceback |
| **Detection Analysis** | time series, residual heatmap, fusion chart, comparison and agreement tables, Drone Inspector | "How to read this" under each chart |
| **Experiments** | tabs Calibration / Sweeps / Evasion / Propagation | a runtime estimate (≈ 0.25 s per trial) before each job; results kept for the session; propagation shown at centre_gain 0 **and** 0.5 side by side (D1) |
| **Recovery** | KPIs, recovery map, attack vector over time, multilateration repairs, before/after errors | if every drone is flagged, explains why multilateration has no trusted neighbours (F1) and points to the MDS estimate |
| **Incidents** | filters, list, alarm timeline, explanation, detector details, compare two | the timeline re-runs the stored configs (deterministic, R2) |
| **Evidence Centre** | build, download, VERIFY, tamper test (red button), custody log, key fingerprint | verification is logged in the custody chain; the tamper test works on a copy and highlights the flipped byte |
| **Methodology** | 12 equations (each naming its source file), assumptions, blind-spot table, F1–F8, D1–D4, credits, limitations | the blind-spot table is built from the scenario presets themselves |

---

## 11. Changes to earlier phases (and why)

- `database/storage.py`: `find_incident_by_run_key()`, so an identical run
  reuses its incident.
- `forensics/incident.py`: the `run_key` field, `save_run()` and
  `open_incident()`. `forensics/evidence_package.py`: `package_incident()`.
  **Why:** the CLI and the app now share one code path, so both produce
  identical incidents and packages.
- `cli.py`: `investigate` on an already-investigated run reports the
  existing package instead of building a duplicate.
- `reports/pdf_report.py`: attacked drones in the flight-path figure are
  forest, so red stays strictly for alarms.

---

## 12. Tests (`tests/test_phase5.py`, 41 tests; 186 in the whole suite)

- **Design system:** config.toml matches the palette; no hex colours in
  `style.css` or `hero.html`; every CSS variable used is defined; stickers
  are valid SVG; hero placeholders all filled; all builders escape text.
- **Figures:** templates registered; the map animates every 5 steps and
  supports all modes; every figure builds; the map zoom fits the flight.
- **Data shaping:** events newest first and honest wording; counts, rows,
  inspector; agreement matrix shows A silent on a rigid shift.
- **Incidents:** the same run never opens two incidents (app and CLI).
- **The real app (AppTest):** every view loads on a fresh database, and
  exactly the five data pages show an empty state; the quick start then
  works on every page, and the evidence package verifies (PASS) and the
  tamper test turns the copy from PASS to FAIL; invalid Attack Lab settings
  give a friendly error.
- **Browser regressions:** graphics reach `st.html` as images, not `<svg>`.

---

## 13. Checked in a real browser

Tests prove the code runs; they cannot prove it looks right. The app was
driven in headless Chrome at 1440, 1024 and 390 px, and screenshots of every
view are in `docs/screenshots/`. That check found and fixed:

1. **Missing graphics.** `st.html` strips `<svg>`; graphics are now images.
2. **Map zoomed in twice too far** (wrong tile-size constant).
3. **Secondary buttons looked primary.** A CSS specificity issue:
   `.stButton > button` (a class + an element) beats `[data-testid=...]`
   (an attribute counts like a class), so the more specific selector won.
4. **Clipped chart titles.** Streamlit sizes the chart box to the figure's
   exact height, so CSS padding pushed the axis titles out of view.
5. **Navigation collapsing to "6 more".** `width: fit-content` made the
   navigation's overflow measurement circular; it now has a set width.
6. **Invisible slider rails and unstyled thumbs.** Streamlit 1.66 builds
   sliders differently from older versions; the selectors were updated.

Also verified: buttons lift on hover, press into their shadow and ripple
(computed styles under forced `:hover`/`:active`); the hero's motion stops
under reduced motion; the light-show quick start runs end to end and its
evidence package verifies, with the tamper test failing as expected.

---

## 14. Honest limitations of the app

- Streamlit re-runs the page on every interaction, so there are no animated
  transitions between pages.
- Experiments run in one process inside the app; very large runs belong in
  the CLI (`--workers`).
- Sweep and evasion results last for the browser session; calibrations are
  saved to disk.
- The map needs internet for its tiles; the Offline toggle draws a plain
  grid instead.
- The signing key is stored unencrypted on disk; production use would
  require a protected key (passphrase or hardware module).

---

## VIVA NOTES

**Q1. Why are the detectors' results cached, and why is that safe?**
Streamlit re-runs the whole page script on every click, so without a cache
each slider move would re-simulate the flight. The cache key is a SHA-256 of
the full simulation, attack and detector configs including the seed. Our
simulation is deterministic (rule R2), so the same key always means the same
result, and reusing it cannot change any number.

**Q2. Why is opening an incident not inside the cached function?**
Opening an incident writes to the database: it is a side effect. A cached
function is skipped when its result is already stored, so the write would
silently not happen. It therefore runs after the cached call, and
`open_incident()` checks the run key so repeat visits reuse the incident.

**Q3. How do you make sure the app, the hero and the PDF use the same colours?**
All colours are defined once in `core/palette.py`. `ui/theme.py` generates
CSS variables from it, `style.css` and `hero.html` only use those variables,
and the PDF reads the palette directly. Tests fail if a hex colour is typed
into the CSS or the hero, or if `config.toml` disagrees with the palette.

**Q4. Status colours fail contrast as text. How is status shown?**
Every status is a text label ("ALARM", "No evidence") in ink on a lightly
tinted fill; the status colour is only used for the border, dot or bar.
Ink on ivory is 18.6:1, far above the WCAG AA minimum of 4.5:1, and colour
is never the only signal, which also helps colour-blind users.

**Q5. Where does the app show ground truth, and how is that kept honest?**
Only in places labelled as evaluation: the Simulator's "Evaluation view"
toggle, the closest-approach chart, Recovery's before/after table, and the
Attack Lab's detected/false-alarm table. Detectors, fusion and recovery
still only receive ObservedFrames (rule R1); the UI reads truth purely to
grade them.

**Q6. Why does the map animate only every fifth step?**
A 60 s run at dt = 0.1 s has 600 steps. One Plotly frame per step makes the
figure large and the browser slow. Every fifth step gives 120 frames, which
still looks smooth while keeping the page responsive.

**Q7. How is the map zoom chosen?**
Web map zoom halves the metres per pixel at each level. With MapLibre's
512 px tiles, metres per pixel = 78271.52 · cos(latitude) / 2^zoom. We solve
for the zoom at which the flight's bounding box, plus a 30 % margin, fits
the panel height, and centre on the middle of the box.

**Q8. What happens on a fresh install with an empty database?**
Every page still works. Pages that need a run (Dashboard, Detection,
Recovery, Incidents, Evidence) show a sticker, one sentence and one button,
which runs the quick-start scenario. An automated test loads all ten pages
on an empty temporary database and checks there is no exception.

**Q9. How does the Attack Lab stop invalid settings from crashing the app?**
The configs validate themselves when created (Phase 1), raising
`ValueError` with a clear message. The page builds them inside
`try/except ValueError` and shows "These settings cannot be simulated:
…", so the user sees the reason instead of a traceback.

**Q10. Why does Recovery sometimes say every drone has insufficient neighbours?**
In a swarm-wide attack fusion flags every drone. Multilateration may only
use trusted, unflagged neighbours (fix F1), and none are left. That is the
correct outcome. Recovery for a swarm-wide attack comes from MDS + RANSAC,
which uses radio ranges and anchors, and the page says so.

**Q11. Why is the Home hero in an iframe?**
`st.html` sanitises HTML, removing scripts and SVG animation elements, so
the moving drone and flowing telemetry would disappear. An iframe renders
our own fixed template exactly. It cannot press Streamlit buttons, so the
real buttons sit under it and the hero's link scrolls to them.

**Q12. How is motion kept accessible?**
Motion is short (90 to 400 ms) and only the Home hero loops. Under the
operating system's "reduce motion" setting, CSS sets every animation and
transition to zero and the hero pauses its SVG animations; this was
verified in a real browser.

**Q13. How does the tamper test in the app prove anything?**
It copies the package, flips one bit of one file in the copy and verifies
it. Before the flip every check passes; after it, that file's hash no longer
matches the signed manifest, so verification fails. The original package is
only read, never written, and the page shows the exact byte that changed.

**Q14. Why were tests not enough, and what did the browser check find?**
Tests prove code runs and produces the right data, not that it looks right.
Driving the app in a headless browser found six visual bugs: graphics
removed by the HTML sanitiser, a map zoomed in twice too far, secondary
buttons styled as primary (CSS specificity), clipped chart titles, a
collapsing navigation bar and invisible slider rails. Each was fixed, and
the first two now have regression tests.
