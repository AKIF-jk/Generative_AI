---
name: Restoration Workstation
colors:
  surface: '#10141a'
  surface-dim: '#10141a'
  surface-bright: '#353940'
  surface-container-lowest: '#0a0e14'
  surface-container-low: '#181c22'
  surface-container: '#1c2026'
  surface-container-high: '#262a31'
  surface-container-highest: '#31353c'
  on-surface: '#dfe2eb'
  on-surface-variant: '#b9cacb'
  inverse-surface: '#dfe2eb'
  inverse-on-surface: '#2d3137'
  outline: '#849495'
  outline-variant: '#3b494b'
  surface-tint: '#00dbe9'
  primary: '#dbfcff'
  on-primary: '#00363a'
  primary-container: '#00f0ff'
  on-primary-container: '#006970'
  inverse-primary: '#006970'
  secondary: '#c0c1ff'
  on-secondary: '#1000a9'
  secondary-container: '#3131c0'
  on-secondary-container: '#b0b2ff'
  tertiary: '#d8ffe7'
  on-tertiary: '#003824'
  tertiary-container: '#65f2b5'
  on-tertiary-container: '#006d4a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#7df4ff'
  primary-fixed-dim: '#00dbe9'
  on-primary-fixed: '#002022'
  on-primary-fixed-variant: '#004f54'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#10141a'
  on-background: '#dfe2eb'
  surface-variant: '#31353c'
typography:
  headline-xl:
    fontFamily: Geist
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.03em
  headline-xl-mobile:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Geist
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 12px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-sm: 0.5rem
  margin: 1.5rem
  margin-sm: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style
The design system establishes a high-precision, technical workstation aesthetic tailored for professional digital conservators, forensic imaging specialists, and visual effects engineers. It prioritizes maximum visual fidelity, low cognitive fatigue during extended operational sessions, and surgical control over complex neural restoration pipelines.

The visual direction merges **Technical Minimalism** with functional **Glassmorphic Hud Elements**. Deep slate and carbon canvas backgrounds isolate visual artifacts, while high-chroma neon cyan and electric indigo accents immediately signal computational states, interactive vectors, and threshold markers without polluting the neutral gray balance required for color-critical restoration tasks. Surfaces project an optical-bench feel: disciplined, linear, and meticulously engineered.

## Colors
The palette is calibrated strictly for dark workspaces and high-dynamic-range display workflows.

- **Neutrals & Surfaces**:
  - Canvas Base: `#0D1117` provides deep black-point grounding without complete visual deadness.
  - Layer 1 (Panels & Toolbars): `#161B22` delivers subtle separation from canvas bounds.
  - Layer 2 (Cards, Dropdowns & Overlays): `#21262D` creates distinct interactive grounding.
  - Structural Borders: `#30363D` provides crisp 1px division lines across dense multi-panel grids.
- **Accents**:
  - Primary (`#00F0FF` Neon Cyan): Dedicated to interactive active states, focal viewport split bars, selection handles, and active processing states.
  - Secondary (`#6366F1` Electric Indigo): Governs secondary tools, AI latent space nodes, model pipeline branches, and focused parameter groupings.
  - Tertiary (`#10B981` Emerald Green): Used strictly for validation statuses, inference completions, and parity metrics.
- **Text & Foreground**:
  - Primary Text: `#F0F6FC`
  - Secondary Text: `#8B949E`
  - Muted/Disabled: `#484F58`

## Typography
Typography reflects extreme technical rigor. **Geist** serves as the primary structural and narrative typeface, delivering neutral, geometric clarity with high legibility at micro scales. **JetBrains Mono** anchors telemetry, parameter readouts, tensor dimensionalities, coordinates, and floating HUD badges.

Numerical data must always use tabular figures (`tnum`) to eliminate layout shifts when dynamically updating histogram levels, inference times, and color channel percentages.

## Layout & Spacing
The layout model employs an edge-to-edge, dockable fluid grid optimized for high-density widescreen monitors (1440px to 4K displays).

- **Grid Architecture**:
  - Standard workstation layout features a fixed 56px primary activity bar, collapsible secondary tool/parameter drawers (280px to 360px), and a viewport canvas that fills all remaining horizontal and vertical space.
  - Docked tool windows utilize an internal sub-grid based on a strict 4px base increment, maximizing functional control density without visual collisions.
- **Responsive Adaptations**:
  - **Desktop (1280px+)**: Multi-viewport rendering (A/B side-by-side, quad-view forensic tiles), persistent property sidebars, and real-time floating inspector palettes.
  - **Tablet (768px - 1279px)**: Toolbars collapse into toggleable off-canvas sheets; viewports switch from side-by-side to swipe-split or toggled overlay inspection.
  - **Mobile (<768px)**: Canvas consumes 100% of the screen area; critical parameters dock to a bottom sheet modal; readouts rely on micro-badges.

## Elevation & Depth
Depth is created through subtle luminance staging and frosted refraction rather than diffused drop shadows:

- **Level 0 (Canvas Base)**: `#0D1117`, totally non-reflective.
- **Level 1 (Docked Workspaces & Sidebars)**: `#161B22` bounded by 1px solid `#30363D` borders. No drop shadows.
- **Level 2 (Inspector Overlays & Tooltips)**: Background `#161B22` with 80% opacity, backed by `backdrop-filter: blur(12px)`. Enclosed by a 1px border colored `rgba(240, 246, 252, 0.12)`.
- **Level 3 (Modal Confirmation & Critical Alerts)**: `#21262D` with an ambient glow (`box-shadow: 0 0 24px rgba(0, 240, 255, 0.08), 0 16px 32px rgba(0, 0, 0, 0.6)`).
- **Split Line / Interactive Pivot**: Features an active 1px core of `#00F0FF` with a subtle 4px directional laser blur (`0 0 8px rgba(0, 240, 255, 0.45)`).

## Shapes
The design adopts a **Soft/Industrial (Level 1)** corner profile. Radii are kept intentionally tight (4px on components, 6px on floating panels, 0px on viewport canvas splits) to reinforce the precision-machined, instrument-grade persona. Large pill radii are strictly avoided, as rounded edges waste pixel real estate in high-density parameter panels.

## Components

### Buttons
- **Primary Tool Button**: Solid `#00F0FF` fill with `#0D1117` text, font weight 600. On hover, background shifts to `#38F8FF` accompanied by a localized cyan ambient glow.
- **Secondary / Action Button**: `#21262D` surface, 1px `#30363D` border, `#F0F6FC` text. Hover prompts border transition to `#8B949E` and surface transition to `#30363D`.
- **Ghost Utility Button**: Transparent surface, `#8B949E` icon/text. On hover, background fills with `rgba(240, 246, 252, 0.05)` and text shifts to `#F0F6FC`.

### Sliders (High-Contrast Parameter Adjusters)
- **Track**: 4px height, `#21262D` base with filled active segment in `#00F0FF` (or `#6366F1` for latent parameters).
- **Thumb**: 12px width by 16px height rectangular scrub block, `#F0F6FC` with a vertical 2px central laser notch (`#0D1117`). High hover responsiveness with numeric scrub tooltips displaying in `label-sm` font.

### Inspection Panes (Floating HUDs)
- Translucent backdrop (`rgba(22, 27, 34, 0.85)` + 16px blur) bounded by 1px subtle hairline `#30363D`.
- Header bar includes a monospace technical tag, zoom coordinates (e.g., `X: 1042 Y: 890 [800%]`), and instant channel isolate toggles (R, G, B, Alpha, Mask).

### Side-by-Side Comparison Viewport & Split Bar
- Viewports run zero-gap edge alignment.
- The split divider is a vertical 2px line in `#00F0FF` anchored by an 18px diamond or dual-chevron interactive handle.
- Sub-labels ("ORIGINAL SOURCE" vs "RESTORATION INFERENCE") display in floating frosted badges (`rgba(13, 17, 23, 0.75)`) at top-left and top-right using `label-md`.

### Inputs & Number Fields
- Height 28px, background `#0D1117`, border 1px solid `#30363D`, text font `JetBrains Mono` at 12px. Focused state replaces border with `#00F0FF` and adds a subtle cyan ring offset.

### Chips & Badges
- Low-profile tags (18px height) using `label-sm`.
- Status indicator: Dot in `#10B981` (Ready), `#00F0FF` (Inferring), or `#6366F1` (Cached), set against `rgba(255, 255, 255, 0.04)` fill.
