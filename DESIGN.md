# ReturnWise design direction

## Brand

ReturnWise is a practical decision-support workspace for teams that need to rank return-risk reviews against a limited review capacity. Its voice is measured and specific. The interface treats model output as an estimate and keeps research status visible beside any score.

The visual character is field-notes meets operations desk: warm paper surfaces, dark forest-green controls, compact data tables, and a small parcel-and-return mark. Visuals should clarify the review process, dataset, or model evidence.

## Design Read

Build the static site around local CSV inspection and a capacity-constrained review list. Risk ranking uses probabilities already present in an unlabeled file; hosted Python inference is absent. Keep research limits visible beside the workflow.

- ENERGY: 2. Warm editorial headings with a compact operations desk.
- RHYTHM: 2. Use an open hero, two file-mode panels, an interactive workspace, a research record, and a direct closing action.
- MOTION: 1. Use short focus and hover transitions only; respect reduced-motion preferences.

## Palette

| Role | Color | Use |
|---|---|---|
| Forest | `#164E35` | Primary actions, wordmark, selected navigation |
| Leaf | `#24734D` | Secondary actions and chart series |
| Soft lime | `#A9C978` | Small emphasis marks and priority highlights |
| Warm paper | `#F6F8F4` | Main background |
| White | `#FFFFFF` | Reading surfaces and forms |
| Ink | `#1B2720` | Main text |
| Muted ink | `#526156` | Supporting text |
| Rule | `#C6D0C6` | Dividers and control boundaries |
| Risk amber | `#8A4B08` | Caution state, paired with a label |
| Risk red | `#A12B27` | Error state, paired with a label |

Use forest green for hierarchy and actions, not as a page-wide fill. Keep status meaning in text as well as color.

## Logo

The mark pairs an outlined parcel with a return arrow. It is built as SVG and has an icon-only version for the favicon and compact navigation, plus a wordmark version for the app and site header. The parcel identifies the order; the arrow identifies the review-and-return cycle.

## Typography

Use system sans-serif for controls, tables, and body copy so the dense parts stay familiar and legible. Use Georgia for page-level headings to give the workspace an editorial tone. Use monospace only for file paths, schema fields, and code. The static site uses local system fonts and does not load fonts from a third-party host.

Use sentence case. Page headings should explain the task or decision, not use campaign language. Keep body text at 16 px where the renderer allows it and avoid low-contrast small labels.

## Components and layout

- Keep the six existing work areas: overview, dataset quality, exploratory analysis, risk queue, decision simulator, and research.
- Keep navigation compact and grouped by workflow. Use the Streamlit navigation drawer on narrow screens.
- The overview pairs a concise explanation and next actions with the real workspace state. If data or a usable scoring artifact is absent, show the next step in that same area.
- The two mode panels distinguish real file states: inspect any CSV, then rank only when valid supplied scores are present. Capacity appears in a separate full-width callout.
- The static header uses a translucent fill and small backdrop blur so its links remain distinct while it stays pinned during scroll. No other surface uses glass styling.
- Use bordered white reading surfaces for related controls and evidence. Avoid shadows as decoration.
- Give the illustrated queue preview one soft shadow so it reads as the primary workspace example; other cards use borders. The small capacity ring uses a conic gradient to show its illustrative fraction.
- Use data tables for inspectable transactions and quality findings. Name every table and chart for assistive technology.
- Keep the established Streamlit spacing rhythm. Let columns collapse naturally; do not compress table labels to preserve a desktop grid.
- On the static Netlify site, use a restrained content width, a mobile menu, and transaction rows that become stacked cards on narrow screens.

## Interaction states

Every control must have a visible label, keyboard focus, and a clear result. Uploads show validation, loading, success, and error states. Risk scoring and capacity controls must use actual uploaded rows and model probabilities. Demo probabilities are labeled as illustrative and remain separate from research results.

Search, capacity selection, and downloads must work on the displayed queue. A failed scoring request must identify what is missing and the next step. No control should imply production scoring when only the historical research artifact is available.

## Responsive behavior

Design for a fluid range rather than fixed device widths. The Streamlit drawer handles narrow navigation. Fixed metric grids stack as their content needs. On the static site, navigation wraps into a menu; feature sections reduce from three columns to two and then one; transaction rows become readable cards; form controls remain at least 44 px tall; and no section may require horizontal page scrolling.

## Visual principles

1. Show the review decision before secondary analysis.
2. Use real dataset and experiment state when available. Label historical or exploratory evidence at the point of display.
3. Use diagrams and interface previews only to explain the return-review workflow.
4. Prefer whitespace, rules, and alignment over shadows, glow, gradients, decorative grids, and excess badges.
5. Preserve uncertainty: a probability is not a certain outcome, and prioritizing review is not evidence that the return will be prevented.

## Smart Retail workspace direction

Smart Retail extends the forest-and-paper palette into a shop operations workspace. The page voice is direct Indonesian; financial values identify whether they are revenue, cost, or estimated gross profit. Smart Retail has a separate local SQLite database and does not reuse ReturnWise research data or its model.

- ENERGY: 2. Keep the workspace calm and workmanlike, with a clear sales register as the most distinctive page.
- RHYTHM: 2. Use a concise operational overview, dense but readable record tables, a task-focused POS, and analysis pages that pair results with calculation notes.
- MOTION: 1. Use Streamlit's native focus and interaction states; no decorative animation.
- Keep the existing forest green, warm off-white, soft gray rules, and restrained amber/red status colors. The local theme supplies the palette and serif page headings, while system sans remains familiar in controls and tables.
- Prefer Streamlit's native navigation, forms, tables, metrics, and charts. Native components preserve keyboard operation and keep the interface maintainable without a custom CSS layer.
- Keep overview metrics to the decisions a shop owner needs first. Inventory and transaction records stay tabular because users need to scan and compare exact values.
- The POS gives product search and cart editing a direct path, with amount due, paid amount, and change shown together before checkout.
- Every operational count and chart comes from stored retail records. Empty stores show an onboarding state; forecasts and recommendations state their data requirements and calculation basis.
- On narrow screens, favor stacked controls and contained table scrolling over shrinking labels or clipping the page.
