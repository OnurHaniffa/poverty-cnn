# Deck Design System — LOCKED 2026-06-21

The presentation is built as **HTML/CSS slides rendered with headless Chrome** (not pptx, not Google
Slides). Every slide is self-verified by screenshot before the user sees it — what Chrome renders IS
what the user presents. Reference build: `deck/01_title.html` → `deck/assets/01_title.png`.

## Why HTML
python-pptx + Google-Slides-API both produced generic output AND a render gap (LibreOffice ≠
PowerPoint/Keynote). HTML gives full design control + WYSIWYG self-verification.

## Aesthetic: "cinematic earth-observation"
Rich, designed, atmospheric — NOT minimalist (user rejected flat/editorial as "boring"). NOT big
low-res satellite photos (they pixelate — keep imagery small/crisp or use vector graphics).

## Palette
- Background: layered radial **gradient mesh** on near-black —
  `radial-gradient(58% 75% at 80% 26%, rgba(24,180,205,.26), transparent 60%)`,
  `radial-gradient(54% 64% at 16% 88%, rgba(98,72,225,.22), transparent 62%)`,
  `radial-gradient(38% 46% at 96% 92%, rgba(222,58,140,.12), transparent 60%)`,
  `linear-gradient(170deg,#060e1a,#040a13 60%,#03070e)`
- Accent: cyan `#18cfe6`, glow cyan `#46e3f0`, teal `#34d6c0` (used in `linear-gradient(96deg,#7af0ff,#34d6c0)` for gradient text + rules)
- Body text: `#9fb4c6` (slate) · headline white `#fff` · mono labels `#5a7588`/`#8fd6e2`
- Atmosphere accents: violet `#6248e1`, magenta `#de3a8c` (background only)

## Type (Google Fonts)
- **Display:** Bricolage Grotesque (800/700) — headlines, big numbers
- **Body:** Hanken Grotesk (400/500/600)
- **Mono:** JetBrains Mono (500) — eyebrows, page markers, data chips, coordinates

## Recurring components
- **Data-globe** (`deck/assets/_globe.svg`): wireframe sphere (parallels/meridians), plotted cluster
  dots with glow filter, atmospheric rim, orbit ellipse + satellite glyph. Hero of dark slides.
- **Eyebrow:** mono, letter-spacing 6px, cyan, glow.
- **Rule:** 96×5px cyan→teal gradient with glow.
- **Page marker:** `NN / 28` mono, top-left or top-right.
- **Data chip:** rounded pill, mono, cyan border, semi-transparent fill (e.g. "23 COUNTRIES · 36,090 VILLAGES").
- **Grain:** SVG feTurbulence data-URI, opacity .05, mix-blend overlay.

## ONE scheme — LIGHT everywhere (locked 2026-06-21, do NOT use dark)
EVERY slide is **light**: near-white `#eef1f4` bg with a faint cyan/violet mesh, navy ink `#0f2636`,
cyan/teal accents (`#16b9d0` → `#1aa985`), white cards with soft shadows + a cyan top/left border,
JetBrains-Mono labels in `#1296ad`. Title + dividers use a **navy/teal wireframe data-globe** on the
same light canvas (see `build_light_deck.py:make_globe`). The dark gradient-mesh theme is RETIRED —
the user wants no dark slides at all. `lib.py` still defines DARK_THEME but it must not be used.
Reference slides: `slides/s01_title.png`, `slides/s02_frame.png`, `slides/s08_pca.png`.

## Render command
```
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
  --force-device-scale-factor=1 --window-size=1920,1080 --virtual-time-budget=4500 \
  --screenshot=OUT.png file://ABS/PATH/SLIDE.html
```
Slides are 1920×1080. Always Read the screenshot to self-verify before showing the user.

## Verify EVERY batch (mandatory) — `deck/verify.js`
After rendering any slides, run `node verify.js slides/*.html` (puppeteer-core driving system Chrome).
It reads each element's real `getBoundingClientRect()` and flags **overlaps** + **off-canvas** elements —
catches collisions the downscaled screenshots hide. It ignores backgrounds (`.grain`/`.gl`/svg) and
decorative no-class elements (e.g. stacked layers). Output must be all `OK` before showing the user.
(No Playwright MCP is connected; this is the equivalent.) User rule 2026-06-22: verify 100% every time.
