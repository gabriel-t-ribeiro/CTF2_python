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
