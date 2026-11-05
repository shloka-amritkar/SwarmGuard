############################################################################
# PHASE 7 — MAP FIXES + "GROUND TRUTH" ORIGINAL DESIGN SYSTEM
# A full visual re-theme of the existing app. Keep ALL functionality,
# views, data flow, tests and component function signatures. Change only
# visuals, motion, the Home hero, the navbar and the map behaviour.
############################################################################

GOAL
Give SwarmGuard an original, elegant, highly professional identity of its
own. It must not resemble any existing product or common "AI dark mode"
(no black/purple/blue neon, no glowing red). It must feel like serious
security and forensics software through PRECISION (coordinates,
crosshairs, survey lines, hashes, case numbers, radar), not through
colour clichés. Warm, calm, confident, premium.

CONCEPT: "GROUND TRUTH" — topographic survey meets forensic case file
- Survey maps: sand-beige paper, olive contour lines, espresso-brown ink,
  coordinate ticks, crosshairs, measured precision.
- Forensic case files: case numbers, folder tabs, evidence stamps, seals,
  hashes, chain of custody.
- Instruments: an amber-phosphor radar scope (like vintage amber
  instrument displays) inside warm charcoal panels.
The name fits the project: "ground truth" is exactly what the detectors try
to recover.

############################################################################
# STEP 0 — PREPARATION (do first, commit after)
############################################################################
0.1 Update the UI DIRECTION section of CLAUDE.md to summarise this design.
    It replaces all earlier UI directions. Update PROJECT STRUCTURE if
    files change.
0.2 Remove third-party reference images from the repository (e.g.
    docs/design-refs/*) and add docs/design-refs/ to .gitignore. They are
    other companies' copyrighted content and must never be published.
0.3 Update the shared palette module (used by ui/theme.py and the PDF
    report) to the palette below. Regenerate one example PDF and check it.
0.4 Commit to main.

############################################################################
# PART A — MAP FIXES (implement and verify before the re-theme)
############################################################################
Problem 1: dragging the map draws a zoom box instead of moving the map.
Problem 2: the camera is fixed, so the swarm flies out of view.

A1 Interaction
   - layout dragmode = "pan" for every map figure.
   - st.plotly_chart config: scrollZoom true, displaylogo false,
     modeBarButtonsToRemove: select2d, lasso2d, zoom2d. Keep reset view.
A2 Camera mode (a small segmented control above every map):
   - "Follow swarm" (default): each animation frame sets layout.map.center
     to the swarm's GPS-reported centroid at a fixed close zoom (tuned so a
     10-drone formation of radius 8 m fills about half the view; the
     20-drone light show about two thirds). Drones never leave the view.
   - "Whole flight": centre and zoom computed once from the bounding box of
     every position in the run, with ~15% padding.
   - Switching modes keeps the current animation time.
A3 Static (non-animated) map views use "Whole flight" automatically.
A4 Verify in the headless browser: single_hijack and light_show_pune in
   both modes; dragging pans; scroll zooms; reset view works.

############################################################################
# PART B — DESIGN SYSTEM
############################################################################

BANNED (anywhere in the app or PDF)
- neon, glow, bloom, gradients used as decoration, glassmorphism, blurred
  drop shadows
- black/near-black UI backgrounds, purple, saturated blue, saturated red
  outside true alarm states
- emojis; cartoon or mascot graphics; stock icon packs other than
  Streamlit Material icons
- the old elements: floating pill navbar, text flowing along curved
  paths, waveform pill, lavender, Instrument Serif, Bricolage Grotesque,
  Inter, JetBrains Mono, forest green #0B4D40
- any element still showing Streamlit's or Plotly's default look

B1. PALETTE (shared palette module; ui/theme.py re-exports it)
Paper and ink:
  sand       #F3EEE3  page background (survey paper)
  card       #FBF8F2  cards, inputs (warm white, never pure white)
  vellum     #ECE5D6  secondary surfaces, table headers, hover rows
  espresso   #1E1A16  primary text, outlines, hard shadows
  ink_2      #4A433B  secondary text
  stone      #8C8579  captions, ruler labels, disabled
  rule       #DDD5C4  hairlines inside cards
  contour    #E4DCCB  background contour lines and grids
Brand and interaction:
  olive      #3F4F24  brand: active nav, links, key headings, selections
  olive_2    #566B33  hover state of olive elements
  sage       #DCE3CC  soft olive tint: selected rows, pills, focus fills
  brass      #D9A441  primary buttons
  brass_2    #C68F2E  button hover/pressed, ripple
  terracotta #B5552F  secondary accent (sparingly: illustration details,
                      Detector D)
Instrument panels (the only dark surfaces):
  charcoal   #23201C  warm charcoal panel background
  charcoal_2 #2D2924  raised rows inside panels
  char_line  #3B362F  panel borders, gridlines, range rings
  char_text  #EFE8DA  panel text
  char_mute  #A69D8E  panel secondary text
  amber      #F2B24C  radar sweep, flight trails, neutral traces on panels
Status (always paired with a text label, never colour alone):
  NORMAL #5B7F3A (olive green) · SUSPICIOUS #D28A1E (amber) ·
  ALARM / COMPROMISED #B3261E (brick red — the ONLY red in the app) ·
  NO_EVIDENCE / INSUFFICIENT_EVIDENCE #8C8579 (stone)
Detectors (fixed everywhere):
  A #4B6E8E (dusty steel) · B #2F7D6D (deep teal) · C #B8862B (ochre) ·
  D #B5552F (terracotta)
Contrast: all text meets WCAG AA on its background (check espresso on
sand/card, char_text on charcoal, espresso on brass).

B2. TYPOGRAPHY (Google Fonts, SIL Open Font License, with fallbacks)
  Fraunces (opsz on; 500–600; italic for emphasis): hero and page titles
  IBM Plex Sans (400/500/600): body, navigation, tables, buttons
  IBM Plex Mono (400/500): KPI numerals, coordinates, hashes, case numbers,
    timestamps, ruler labels, eyebrows
Scale:
  hero 60px Fraunces 500 (38px under 900px), line-height 1.05
  page title 38px Fraunces 500; one italic word allowed per title
  section title 20px Plex Sans 600
  eyebrow 11px Plex Mono 500 uppercase, letter-spacing 0.16em, stone
  body 15px/1.6 Plex Sans; table 14px; captions 13px stone
  KPI value 34px Plex Mono 500 tabular numerals; KPI label 11px eyebrow
Page titles follow a pattern: eyebrow ("02 · DETECTION ANALYSIS"), then a
Fraunces title with one italic word ("How each detector *saw* the attack").

B3. LAYOUT, SHAPES, SHADOWS
- Max content width 1320px; 24px gutters; 8px spacing grid; 56px between
  page sections.
- Cards: card fill, 1.5px espresso outline, radius 14px. Feature cards add
  a hard shadow 4px 4px 0 espresso. Dense data cards: outline only.
- Instrument panels: charcoal fill, 1px char_line border, radius 14px,
  no shadow.
- Inputs: card fill, 1.5px espresso outline, radius 8px; focus 2px olive.
- Pills: radius 999px, 1px espresso outline, tinted fill (sage, or status
  colour at ~20%), Plex Mono 11px uppercase.

B4. SIGNATURE MOTIFS (what makes it ours)
M1 Topographic contours: the page background shows very faint olive-beige
   contour lines (an inline SVG of smooth concentric organic curves in
   contour colour, ~6% visual weight), fixed behind content. Different
   contour arrangement per page section is not needed; one subtle
   background is enough.
M2 Crosshair corners: feature cards and instrument panels carry small
   registration marks at all four corners (CSS pseudo-elements, 10px
   L-shaped ticks; espresso on cards, char_line on panels).
M3 Coordinate rulers: the Home hero has tick rulers along its top and left
   edges with Plex Mono labels (18.510°N … 18.530°N, 73.845°E … 73.865°E).
M4 Amber radar: instrument panels with maps or the hero radar use amber
   for sweep and trails; range rings in char_line.
M5 Hash ribbon: a thin espresso band (sand text, Plex Mono 12px) slowly
   scrolling real SHA-256 hashes from the newest evidence manifests,
   prefixed with case numbers ("SG-20261105-001 · 9f2c…e41a ·").
   Fallback: hashes of the demo scenario configs. Pauses on hover.
M6 Case files: incidents appear as cards with a folder tab on the top edge
   holding the case number in Plex Mono, a "filed" timestamp, and a
   status stamp.
M7 Evidence stamps: rubber-stamp marks with rough ink edges (SVG
   feTurbulence + feDisplacementMap filter), slightly rotated:
   "VERIFIED" in olive on verification PASS; "TAMPERED" in brick red on
   tamper-test FAIL; "FILED" in stone on new incidents. Each stamp shows
   the case number and UTC time in tiny mono text inside its border.
M8 Survey marks: section dividers are a thin rule with a small crosshair
   in the middle and a coordinate label on the right (e.g. "§ 03 ·
   18.5204°N").

B5. NAVBAR (replaces the floating pill)
Full-width top bar on sand, 1.5px espresso bottom border, 64px tall.
- Left: a small radar-scope mark (circle, two range rings, sweep line, in
  espresso) + wordmark "SWARMGUARD" in Plex Mono 500, letter-spacing
  0.14em.
- Centre: page links in Plex Sans 500; active link olive with a 2px
  underline that slides between links (240ms).
- Right: live readout in Plex Mono 12px stone:
  "18.5204°N 73.8567°E · UTC hh:mm:ss" (seconds tick), then the current
  scenario name and a status pill.
- Use st.navigation(position="top") if supported, styled to match;
  otherwise a custom bar plus a styled sidebar on narrow screens.

B6. BUTTONS
Primary: brass fill, espresso text (Plex Sans 600, 14px), 1.5px espresso
  border, radius 8px, hard shadow 3px 3px 0 espresso.
  hover: translate(-1px,-1px), shadow 4px 4px 0, fill brass_2
  active: translate(3px,3px), shadow 0 (presses into its shadow), a ripple
          circle in brass_2 expanding from the centre (CSS only)
  focus-visible: 2px olive outline, offset 3px
Secondary: card fill, same outline, shadow and motion.
Quiet/text button: olive text with an underline that grows on hover.
Destructive (tamper test only): brick fill, sand text, same mechanics.
At most one primary button per screen.

B7. MOTION
Easing cubic-bezier(0.2, 0.8, 0.2, 1). Durations: 90ms press, 140ms
hover, 240ms entrances, 280ms stamps, 400ms count-up and draw-on.
- Page entrance: content blocks fade in and rise 8px, staggered 40ms
  (max 6 steps).
- KPI count-up from 0 (CSS @property integer counter; static fallback).
- Tabs: the indicator slides between tabs in olive.
- Verification: each PASS row draws its checkmark; FAIL rows shake once
  (240ms); then the stamp lands (M7): scale 1.6 -> 1, rotate to -8deg,
  opacity 0 -> 1 with slight overshoot, and the card nudges 1px ("thud").
- Toasts restyled as small case-file cards.
- All motion disabled under prefers-reduced-motion.
- Honest note for the docs: Streamlit reruns on interaction, so there are
  no animated transitions between pages; entrance animations provide the
  smooth feel.

B8. HOME HERO (rewrite assets/hero.html entirely)
Desktop: two columns inside the coordinate rulers (M3). Mobile: stacked.
Left column (on sand):
  eyebrow "SWARMGUARD · GPS INTEGRITY & DIGITAL FORENSICS"
  headline (Fraunces): "Where the drones" / italic "really are."
  subtitle (Plex Sans, 2 lines): "Simulates GPS spoofing on a drone
    swarm, catches it with three independent detectors, and seals the
    evidence so it can't be quietly changed."
  real st.button CTAs below the iframe, styled identically:
    primary "Run a scenario", secondary "How it works"
  three small mono stats under the CTAs, filled from the project:
    "239 tests · 3 detectors · Ed25519-signed evidence" (compute the test
    count at build time or keep it in a constant updated by the CLI)
Right column (instrument panel with crosshair corners):
  An amber radar scope: three range rings, bearing ticks every 30°, a slow
  rotating sweep line with a faint trailing wedge (8s per rotation), and a
  formation of 10 drone blips that brighten as the sweep passes. Every ~10s
  one blip separates into two points, its TRUE position (filled amber dot)
  and its SPOOFED GPS position (hollow dot), joined by a dashed line that
  grows; after ~1.5s a small brick-red label "ALARM · motion check"
  appears beside it, holds, and fades; the blip then rejoins. Bottom-left
  of the panel: a tiny mono readout "NIS 2.1 → 48.6" that updates during
  the event.
Below the hero: the hash ribbon (M5).
Render via streamlit.components.v1.html (iframe), fonts loaded inside,
transparent background; pause everything under reduced motion.

B9. VIEW-BY-VIEW TOUCHES (keep all existing content and behaviour)
Home: hero; three feature cards (Detect / Recover / Prove) with line-art
  illustrations; "Why one detector isn't enough" card with the
  cancellation equation; architecture diagram restyled in palette colours;
  four scenario cards with crosshair corners.
Dashboard: eyebrow + title; KPI row; swarm map in an instrument panel with
  the camera control; drone table; event feed in an instrument panel; a
  slim hash ribbon; pattern banner as a card with a brick or amber left bar.
Swarm Simulator: map panel with camera control, evaluation view and
  offline toggles; a legend card using survey-style symbols.
Attack Lab: controls in outlined expanders with olive section labels; one
  brass RUN button; results table; expected-vs-actual shown as sage/brick
  pills with text.
Detection Analysis: charts on cards with the light template; Drone
  Inspector styled as a case-file side card.
Experiments: tabs with sliding olive indicator; progress bars in olive on
  vellum tracks.
Recovery: recovery map panel; attack-vector chart on a card.
Incidents: case-file cards (M6) with FILED stamps; two-incident comparison
  side by side.
Evidence Centre: case-file layout: header with case number and filed time;
  verification table in an instrument panel; VERIFIED / TAMPERED stamps
  (M7); custody chain as a vertical timeline with survey-mark nodes.
Methodology: textbook-like layout: Fraunces section titles, equations in
  outlined cards, margin notes in stone; credits updated (see B11).

B10. CHARTS AND MAP
Templates (rename): "groundtruth_light" for charts on cards: transparent
  background, Plex Sans, espresso axis lines 1px, faint dashed grid in
  rule, stone tick labels, 2.25px traces, threshold lines dashed espresso
  with right-aligned labels, legend horizontal above, transitions 300ms.
  "groundtruth_panel" for instrument panels: charcoal background,
  char_line grid, char_text labels, amber neutral trace.
Swarm map (instrument panel): default basemap "carto-positron" desaturated
  feel works poorly on charcoal, so use "carto-darkmatter" inside panels;
  drones are circles with a 1px card-colour outline and status fill;
  trails in amber fading over the last 20 positions; anchored drones as
  diamonds; optional faint range rings (10 m, 20 m) around the swarm
  centroid in char_line; formation links amber at 15% opacity.
Heatmaps: card -> vellum -> brass -> brick.
Bars and lines use detector colours; status colours only for status.

B11. ILLUSTRATIONS AND CREDITS
- Redraw stickers as refined survey-style line art: espresso 1.5px strokes,
  fills card/vellum/brass/sage, no tilt. Subjects: quadcopter (top and
  side), satellite, radar dish, GNSS constellation, shield with check,
  padlock, fingerprint, sealed document, crosshair, survey tripod.
- Methodology credits: describe the UI as SwarmGuard's original "Ground
  Truth" design system; keep one line noting general influence from
  neo-brutalist web design; remove references to specific companies'
  websites. Keep all research credits (R6) unchanged.

############################################################################
# FINISH
############################################################################
- Check every view at 1440px, 1024px and 390px in the headless browser,
  on an empty database and after the light_show_pune quick start.
- Replace all screenshots in docs/screenshots/ (include one of the hero
  mid-animation with the ALARM label visible, and the Evidence Centre with
  the VERIFIED stamp).
- Write docs/notes/phase-7-explained.md: the design rationale (why warm
  survey colours instead of "hacker" dark mode, what each motif means),
  the map fixes, and 6 viva questions with answers.
- Run all tests; commit to main.

ACCEPTANCE CHECKS
- Map: dragging pans, scroll zooms, reset works; in "Follow swarm" drones
  never leave the view; "Whole flight" shows the entire light-show loop.
- None of the banned elements remain (search the code for the old font
  names, lavender/forest hex values, textPath, waveform).
- Only brick red appears for alarms; no black or blue UI surfaces.
- Hero radar sweep and the true-vs-spoofed blip event run smoothly and
  stop under reduced motion; the hash ribbon scrolls real hashes.
- VERIFIED stamp lands on PASS; TAMPERED stamp lands on FAIL.
- Regenerated PDF uses the new palette and fonts.
- All tests pass.
############################################################################