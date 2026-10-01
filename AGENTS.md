# CTF2 Project Guidelines (Root Workspace)

## Consistent Nomenclature & Terminology Rules (Universal Principle)
- **Universal Rule of Single Identification (Zero Synonyms / No "Elegant Variation")**:
  - In all scientific writing, documentation, code docstrings, and comments, **never use synonyms** for any concept, parameter, method, component, or physical entity once introduced.
  - Every defined concept (hardware components, mathematical variables, algorithmic steps, optical metrics, boundary conditions, physical phenomena) must maintain a strict **1:1 lexical mapping**: use the exact same canonical term every single time it is referenced.
  - Alternating terms (introducing stylistic synonyms) creates semantic ambiguity, misleading reviewers and developers into thinking a different entity, modified geometry, or alternate formulation is being discussed.
- **Mandatory Canonical Anchors**:
  - **Tubo**: Always use **"tube"** (never "pipe", "duct", "channel", "conduit", etc.). The geometry is unequivocally circular and hollow.
  - **Bancada experimental**: Always use **"experimental setup"** (never "test rig", "test bench", "facility", "experimental apparatus", etc.).
  - **All other concepts**: Apply strict 1:1 consistency across all models, variables, algorithms, and procedures.

## Mathematical Formulations & Notation Rules
- **Explicit Definition of All Formula Components**:
  - Every mathematical formula and equation appearing in any project document must have **all of its components, variables, indices, subscripts, and constants explicitly defined and explained** immediately following the equation (e.g., in a "where $x$ represents..., $y$ is..." clause or explanatory sentence).
  - No symbol should ever be left undefined or assumed to be obvious to the reader.
- **Visual Grounding via Figures / Schematics**:
  - Geometric, physical, and optical variables (e.g., coordinate frames, semi-axes $a, b, c$, chord length $L_{\text{chord}}$, tilt angle $\theta$, defocus distance) should preferably be anchored with an explanatory schematic/figure.
  - Visual grounding eliminates ambiguity, enhances clarity for reviewers, and ensures rigorous reproducibility.

## Project Structure & Sub-Repository Guidelines
- **Nested Repositories**:
  - `ctf2_article/`: Independent Git repository for the scientific manuscript. Refer to [`ctf2_article/AGENTS.md`](file:///Users/gabrielribeiro/Library/CloudStorage/GoogleDrive-gribeiro.gr933@gmail.com/O%20meu%20disco/Estagio/CTF2_python/ctf2_article/AGENTS.md) for LaTeX-specific rules (e.g., `\autoref`, subfigure spacing, build artifact handling).
  - `ctf2_report/`: Independent Git repository for the internship report. Refer to [`ctf2_report/AGENTS.md`](file:///Users/gabrielribeiro/Library/CloudStorage/GoogleDrive-gribeiro.gr933@gmail.com/O%20meu%20disco/Estagio/CTF2_python/ctf2_report/AGENTS.md) for report-specific conventions.
- **Python Codebase (`CTF2_python`)**:
  - Maintain physical SI units throughout calculations and pipeline outputs (e.g., $\text{m}$, $\text{m}^3$, $\text{m/s}$, $\text{s}$).
  - Reflect canonical terminology in variable naming (e.g., prefer `tube_radius`, `tube_diameter` over `pipe_radius`).

## Scientific Figures, Vector Schematics & Diagram Styling Rules
- **No Overlap Principle (Zero Visual Collisions)**:
  - Text cards, equation boxes, annotations, and callouts must never straddle, collide with, or be impaled by canvas borders, container frames, plotted curves, or coordinate axes.
  - Multi-panel or sequence schematics (e.g., inter-frame tracking transitions) must feature wide, dedicated inter-container clearance gaps ($\ge 30\%$ of container width) so that connecting status cards sit entirely in clear space without intersecting frame boundaries.
  - Dynamic trajectory arcs, connection arrows, or curves must never be severed, covered, or obscured by overlying annotation cards. Place explanatory text cards strictly above the curve peak or below the trajectory envelope.
- **Engineering Dimensioning Standards & Side-Pull Leader Lines**:
  - Linear dimensions and physical lengths (e.g., chord lengths $L_{\text{chord}}$, bubble depth semi-axes $c$, search windows) must follow standard technical drawing conventions (ISO/ASME drafting standards):
  - Project dotted extension lines (`ls=":"`) outward from the exact measurement extremities into clear canvas space.
  - Draw a parallel double-headed arrow (`arrowstyle="<->"`) terminating precisely at the extension lines.
  - **Side-Pull Leader Dimensioning (Constrained Spaces)**: Whenever the space between or directly beneath extension lines is narrow, crowded, or would collide with underlying geometry, centerlines, coordinate axes, or other annotations (e.g., vertical $\Delta y$ or tube axis):
    - Do *not* squeeze or center the label card between or directly under the extension lines.
    - Extend the horizontal dimension line outward past the outer extension line into adjacent clear canvas space as a continuous solid leader line (`ax.plot([x_ext, x_leader], [y_dim, y_dim])`).
    - Anchor the dimension label card to the side, directly aligned with the leader line (`ha="left", va="center"`), guaranteeing zero visual collisions with the primary figure geometry.
- **Clearance Around Gradient Curves & Shaded Regions**:
  - Never place text boxes directly inside narrow shaded regions where curves have steep gradients (e.g., asymptotic near-wall blind zones).
  - Place descriptive cards in adjacent open canvas space to the side of the shaded region, using a horizontal pointer arrow directed into the zone.
  - Always maintain a clear margin ($\ge 10\text{--}15\%$ of axis range) from any plotted curve.
- **Matplotlib Mathtext Formatting**:
  - Never use raw LaTeX macros like `\textbf{...}` or `\&` outside math mode; use standard mathtext (e.g., `$\mathbf{(a)}$ Title & Subtitle`).
  - Use supported mathtext operators (`\leq`, `\geq`, `\to`, `\mathrm{-}`, `\tilde{...}`).
