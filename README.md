# From Broken Trajectories to Broken Pixels

**A visual research essay**

> *When Morse geometry illuminates the optimization landscape of neural image reconstruction*

Live site: **https://khadiijatu.github.io/morse-geometry-ct/**

---

## What this is

This is a self-contained, interactive visual essay exploring the conceptual and
mathematical bridge between:

- **Morse-Smale gradient flow theory**, specifically, the stratified structure of
  stable and unstable manifold closures, as studied in the author's MSc thesis
  (Aix-Marseille University, 2020, supervisor: Prof. D. J. A. Trotman)
- **Neural CT image reconstruction**, specifically, Implicit Neural Representations
  and the geometry of their optimization landscapes

The essay is a companion to two papers in preparation:
1. *Conical Models and Whitney Regularity for Stable Manifold Closures in Morse-Smale
   Gradient Flows: A Unified Proof* (arXiv preprint, math.DG / cs.CV)
2. *Geometric Optimization in Differentiable Rendering: A Morse-Theoretic Perspective*

---

## Structure

| Section | Content |
|---------|---------|
| **01 — The Morse Landscape** | Interactive: place critical points, watch gradient flow organize |
| **02 — Broken Trajectories** | Animated compactification of moduli spaces |
| **03 — Stratified Closures** | Whitney stratification diagram, animated |
| **04 — The Conceptual Bridge** | CT reconstruction as gradient flow — side-by-side comparison |
| **05 — Neural Representations** | What INRs are; live simulated training animation |
| **06 — The Open Question** | Precise mathematical statement of the research programme |
| **07 — About & References** | Context, author, bibliography |

---

## Technical notes

- **Zero dependencies** except Google Fonts and MathJax (both CDN, no install needed)
- All diagrams and animations are pure HTML5 Canvas + CSS
- MathJax renders all mathematical notation
- Fully responsive (mobile/tablet/desktop)
- No build step required: `index.html` is the entire site

---

## Running locally

```bash
git clone https://github.com/khadiijatu/morse-geometry-ct.git
cd morse-geometry-ct
# Open index.html in any modern browser, or:
python3 -m http.server 8000
# Then visit http://localhost:8000
```

---

## Author

**Khadidiatou Cissé**
