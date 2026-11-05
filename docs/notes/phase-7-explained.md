# Phase 7 explained: map fixes and the "Ground Truth" design system

Phase 7 changed **how the app looks and how its maps behave**. It did not change
what the app computes. The simulator, detectors, fusion, recovery, forensics,
database and every page's content are the same as in Phase 6. If you are asked
"what did Phase 7 add to the science?", the honest answer is: nothing. It made
the results easier to read and the platform look like a serious tool.

---

## 1. Why the design changed

The Phase 5 look ("Ivory + Console") borrowed its mood from two well-known
websites. A security and forensics product should have its own identity, and
it should not look like the usual "hacker" dark mode (black screens, neon
purple and blue, glowing red). Those clichés say "cyber" without meaning
anything.

**Ground Truth** gets its seriousness from **precision** instead of colour:

| Idea | Where it comes from | What it means for SwarmGuard |
|---|---|---|
| Survey maps | Sand-beige paper, olive contour lines, espresso ink, coordinate ticks | The platform measures positions. Its whole job is metres and coordinates. |
| Forensic case files | Case numbers, folder tabs, rubber stamps, chain of custody | Incidents are evidence. They get numbers, are filed, verified and sealed. |
| Instruments | An amber-phosphor radar scope in a warm charcoal panel | Live, operational data (maps, feeds, verification) lives in "instrument panels". |

The name fits too. "Ground truth" is exactly what the detectors try to recover:
the real positions that GPS no longer tells us honestly.

### The colours and their jobs (`core/palette.py`)

Every colour has exactly one job, so colour itself carries meaning:

- **sand / card / vellum**: paper. These are the page, the cards and secondary surfaces. "card" is never pure white, because warm paper feels calmer.
- **espresso**: ink. It is used for text, outlines and hard shadows.
- **olive**: the ONE brand colour. It marks the active navigation link, links, headings and selections.
- **brass**: the ONE action colour. It fills the primary button (at most one per screen).
- **charcoal + amber**: the only dark surfaces. Instrument panels for the map, the event feed and verification.
- **brick #B3261E**: the ONLY red. It is used for alarms and the destructive tamper test, and nothing else.
- **Status colours** are always shown together with a text label (NORMAL, SUSPICIOUS, ALARM, NO EVIDENCE), because colour alone fails people with colour blindness.

**Contrast check.** Every text colour must reach WCAG AA (4.5 : 1) on its background.
`tests/test_phase4.py` computes it with the WCAG formula. One thing it found:
the spec's grey "stone" (#8C8579) is only **3.2 : 1** on sand, which is too low for
text. So stone is used only for marks (ruler ticks, disabled items, the
NO EVIDENCE dot), and a slightly darker **stone_text** (#6B6459, 5.0 : 1) is used
for caption text. This was reported rather than quietly changing the spec colour.

### The fonts

- **Fraunces** (a serif) is used for page titles and the hero. Each title has one italic word, for example "Swarm *dashboard*".
- **IBM Plex Sans** is used for body text, menus, tables and buttons.
- **IBM Plex Mono** is used for anything measured or identifying: numbers, coordinates, hashes, case numbers and timestamps. Monospace digits all have the same width, so columns of numbers line up.

All three are under the SIL Open Font License. The web app loads them from
Google Fonts. The PDF report cannot use web fonts, so the TTF files are
bundled in `assets/fonts/` with their licence texts.

---

## 2. The map fixes (Part A)

### Problem 1: dragging drew a zoom box

Plotly's default "drag mode" is *zoom*. Dragging draws a rectangle and zooms
into it. On a map, people expect dragging to *move* the map.

**Fix** (`ui/plots.py`, `ui/components.py`):
- Every map figure sets `dragmode="pan"`.
- Maps are shown with `MAP_PLOTLY_CONFIG`, which turns on `scrollZoom` and removes the `zoom2d`, `select2d` and `lasso2d` tools. "Reset view" stays.
- Ordinary charts deliberately do *not* get scroll zoom. Otherwise the mouse wheel would zoom a chart instead of scrolling the page past it.
- `chart()` picks the right config automatically: a figure that pans is a map.

### Problem 2: the swarm flew out of view

The old map computed one camera position for the whole flight and never moved
it. With a close zoom, the swarm left the screen.

**Fix: two camera modes**, chosen with a small segmented control above the map:

**Follow swarm** (the default). In every animation frame, the camera centre is
the swarm's centroid, the average of the GPS-reported positions:

    centre_k = (1/N) * sum_i gps_k,i

Each Plotly animation frame now carries `layout.map.center`, so the camera
travels with the swarm.

The zoom is computed once and kept fixed, so the map does not "breathe":

    metres per pixel at zoom z = 78271.52 * cos(latitude) / 2^z

(78271.52 rather than the familiar 156543.03, because MapLibre tiles are 512 px wide.)

We want the swarm's typical diameter to cover a fraction `fill` of the view:

    z = log2( 78271.52 * cos(lat) * fill * pixels / diameter )

- `fill` = 0.5 for a normal swarm (it fills about half the view).
- `fill` = 0.67 for the 20-drone light show (about two thirds).

A second "safety" zoom is also computed. It makes sure that even the largest
radius ever seen, including a spoofed drone reported 10 m away, fits with a
15 % margin. The smaller (wider) of the two zooms is used. **So no drone ever
leaves the view.**

**Whole flight.** The camera is computed once from the bounding box of every
position in the run:
- the centre is the middle of the box, (min + max) / 2;
- the zoom makes the longest side plus 15 % padding fit the view.

Static, non-animated maps use this mode automatically.

**Why the GPS positions, not the true ones?** The camera shows what an operator
would see. Truth is used only in the labelled evaluation view (rule R1).

**Verified in a real browser** (headless Chromium, Playwright), for
single_hijack and light_show_pune in both modes:
- dragging changes the centre without changing the zoom (pan);
- the wheel raises the zoom;
- "Reset view" restores the start;
- at every sampled frame, every drone's latitude and longitude lies inside the visible map bounds (0 outside, out of 210 and 420 checked positions).

**An honest limitation.** On the Swarm Simulator, the animation plays inside the
browser (Plotly's Play button). Streamlit never learns which frame is showing.
When you switch camera mode, Streamlit reruns the page and the playback starts
again at t = 0. The Dashboard does not have this problem, because its time comes
from a Streamlit slider, which keeps its value. The Simulator could have used a
Streamlit slider too, but then every frame would rerun Python and playback
would look jerky. We chose smooth playback and documented the restart.

---

## 3. The signature motifs (Part B)

| Motif | What it is | How it is built |
|---|---|---|
| M1 contour lines | Faint survey contours behind every page | `ui/theme.py: contour_background()` draws rings with r(θ) = r0·(1 + 0.10 sin 3θ + 0.05 sin 5θ) as an SVG tile in the "contour" colour |
| M2 crosshair corners | Registration ticks at the 4 corners of feature cards and panels | `corner_marks()`: four tiny L-shaped SVGs as four CSS background layers on one pseudo-element |
| M3 coordinate rulers | Tick rulers 18.510°N … 18.530°N and 73.845°E … 73.865°E around the hero | `ui/hero.py: ruler_html()` |
| M4 amber radar | The hero scope; amber sweeps and trails on panels | `ui/hero.py`, `ui/plots.py` |
| M5 hash ribbon | A scrolling band of REAL manifest hashes, e.g. "SG-20261105-001 · 9f2c…e41a" | `storage.list_packages()` → `hash_ribbon_html()`; before any package exists it shows the hashes of the demo scenario configs, marked "config", never invented ones |
| M6 case files | Incidents shown as cards with a folder tab holding the case number | `case_card_html()` |
| M7 stamps | VERIFIED (olive) on PASS, TAMPERED (brick) on FAIL, FILED (stone) on new incidents | `stamp_svg()`: SVG filter feTurbulence (random noise) + feDisplacementMap (wobbles the edges) + a fine noise mask (uneven ink) |
| M8 survey marks | Section dividers: rule, crosshair, rule, "§ 03 · 18.5204°N" | `survey_divider_html()` |

### The navbar

The page links are Streamlit's own `st.navigation(position="top")`. Around them,
`navbar_html()` draws one full-width 64 px sand bar:
- **left:** the radar mark and the SWARMGUARD wordmark;
- **right:** the live readout "18.5204°N 73.8567°E · UTC hh:mm:ss", the scenario and a status pill.

**The clock ticks without JavaScript.** `st.html` removes scripts, and rerunning
the page every second would flicker. Instead, three integer CSS properties
(hours, minutes, seconds) are animated with `steps()`. Each starts part-way
through its cycle using a negative `animation-delay` computed from the server
time when the page is drawn:

    delay_seconds = -s
    delay_minutes = -(60 m + s)
    delay_hours   = -(3600 h + 60 m + s)

### The hero

The radar works like this:
- the sweep turns once every 8 s;
- a blip at bearing b brightens when the sweep reaches it, after t = b/360 × 8 s;
- every 10 s, one blip splits into its true position (filled) and its spoofed GPS position (hollow), joined by a growing dashed line;
- "ALARM · motion check" appears, the NIS readout jumps 2.1 → 48.6, and then the blip rejoins.

It is an illustration of what Detector B does, labelled "simulation" inside the panel.

A point at bearing b and radius r is drawn at:

    x = r sin b,   y = −r cos b

(y is minus, because SVG's y axis points down.)

### Motion, and what Streamlit cannot do

- Durations: 90 ms press, 140 ms hover, 240 ms entrances, 280 ms stamps, 400 ms count-up and draw-on.
- Under `prefers-reduced-motion`, everything stops. The CSS animations are switched off, and the hero's SVG animations are paused with `pauseAnimations()`.
- The UTC clock keeps ticking, because it is information, not decoration.

**Honest note.** Streamlit reruns the whole page on every interaction, so there
are no animated transitions *between* pages. The entrance animation (content
fades in and rises 8 px, staggered 40 ms) provides the smooth feel instead. For
the same reason, the active-link underline slides in when a page arrives,
rather than travelling from the old link to the new one.

---

## 4. Files changed in this phase

**`core/palette.py`**
- New palette; fields renamed (ivory → sand, ink → espresso, forest → olive, console → charcoal …).
- `stone_text` added.
- Every caller was updated.

**`reports/pdf_report.py`**
- Registers the bundled TTFs with ReportLab and matplotlib, falling back to Helvetica/Courier with a logged warning if a file is missing.
- Olive headings and vellum table headers.

**`ui/theme.py`**
- New fonts and tokens.
- Generates the contour background and the corner marks as CSS variables, so `style.css` still has no colour values.

**`assets/style.css`**
- Rewritten for Ground Truth.
- Fixed tab styling for Streamlit 1.66: tabs are now built with React Aria, so the old BaseWeb selectors no longer matched anything.

**`assets/hero.html`, `ui/hero.py`**
- New radar hero with rulers.
- The stats line reads the test count from `assets/test_count.json`.

**`assets/stickers.py`**
- Redrawn survey-style line art (1.5 px strokes, no tilt).
- Old names still work through `ALIASES`.

**`ui/plots.py`**
- Pan, cameras (`camera_follow`, `camera_whole`), range rings, anchor diamonds.
- Templates `groundtruth_light` and `groundtruth_panel`.
- Heatmap scale card → vellum → brass → brick.
- The recovery map moved into a panel.

**`ui/components.py`**
- New navbar, clock, panels, camera control, stamps, case cards, hash ribbon, survey dividers, custody timeline and legend shapes.
- Every existing function signature is kept.

**`ui/state.py`**
- `hash_ribbon_entries()`.

**`database/storage.py`**
- `list_packages()`.

**`cli.py`**
- `count-tests` (writes `assets/test_count.json`).

**Views**
- Numbered eyebrows ("04 · Detection analysis").
- The camera control on the maps.
- Case files on Incidents and Evidence.
- Stamps and the custody timeline on Evidence.
- A textbook layout on Methodology, with updated design credits.
- A retyped Detector D equation: it contained stray form-feed and tab characters.

**Tests**
- `tests/test_phase4.py` and `tests/test_phase5.py` were updated to the new values.
- New `tests/test_phase7.py`: map pan and camera maths, motifs, and a sweep that fails if any retired design element (old fonts, lavender, forest green, curved-path text, waveform, gradients, blur) returns.

---

## VIVA NOTES

**1. Why warm survey colours instead of a dark "hacker" theme?**
Black screens with neon and glowing red are a cliché. They look "cyber" but
carry no meaning, and a forensic tool should look calm and trustworthy. We
chose a visual language that matches what the platform actually does:
measuring positions (survey maps) and handling evidence (case files). Only the
live, operational panels are dark, like the instruments in a control room.

**2. How does "Follow swarm" keep every drone on screen?**
Every animation frame centres the camera on the swarm's GPS centroid. The zoom
is the smaller of two values: one where the typical swarm diameter fills half
the view, and one where the largest radius ever seen still fits with a 15 %
margin. Because the camera always sits on the centroid and the view is at
least as wide as the largest radius, no drone can leave it. We also verified
this in a headless browser at sampled frames.

**3. Why does dragging now pan, and why does only the map get scroll zoom?**
We set Plotly's dragmode to "pan" and removed the zoom-box and selection tools,
so a drag can only move the map. Scroll zoom is enabled only for maps. On an
ordinary chart, the wheel would get "stuck" zooming the chart instead of
scrolling the page.

**4. How is the hash ribbon honest? Could it show fake hashes?**
It reads the manifest SHA-256 of the newest evidence packages straight from
the database. Those are the same hashes that are signed with Ed25519. If no
package exists yet, it shows the SHA-256 of each demo scenario's
configuration, clearly marked "config". It never invents a hash.

**5. A VERIFIED stamp appears. Does that prove the drones were not spoofed?**
No. The stamp means every file still matches the signed manifest, so the
evidence is unchanged since it was signed. It says nothing about whether the
telemetry itself was genuine (rule R4). The TAMPERED stamp after the one-bit
tamper test shows the check is sensitive enough to notice any change.

**6. What happens when you switch camera mode, and why is that a limitation?**
The page reruns. On the Simulator, the Plotly animation restarts at t = 0,
because it plays inside the browser and Streamlit never learns which frame was
showing. The Dashboard keeps its time, because it uses a Streamlit slider.
We chose smooth in-browser playback and documented the restart, instead of
making Python redraw every frame.

**7. How do you make sure the colours stay readable?**
Every text and background pair is checked with the WCAG contrast formula,
(L_light + 0.05) / (L_dark + 0.05), in `tests/test_phase4.py`. AA needs at
least 4.5. Status colours are never used as text: status is espresso text on
a tinted pill. When the spec's caption grey failed (3.2 : 1), we added a
darker text variant rather than ship unreadable captions.
