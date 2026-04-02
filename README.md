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

The essay is a companion to two manuscripts in preparation:
1. *Conical Models and Whitney Regularity for Stable Manifold Closures in Morse-Smale
   Gradient Flows: A Unified Proof*
2. *Geometric Optimization in Differentiable Rendering: A Morse-Theoretic Perspective*

---

## Structure

| Section | Content |
|---------|---------|
| **01 - The Morse Landscape** | Interactive: place critical points, watch gradient flow organize |
| **02 - Connecting and Broken Trajectories** | Animated compactification of moduli spaces |
| **03 - The Stratified Closure** | Whitney stratification diagram, animated |
| **04 - Complete example: Height function on torus** | Explicit computation |
| **05 - The Bridge to CT reconstruction** | CT reconstruction as gradient flow — side-by-side comparison |
| **06 - SIREN loss landscapes: Morse–Bott structure** | Numerical experiment / Morse-Bott structure|
| **07 - Volume growth and Whitney stratification** | Computable certificate applicable to neural network parameter spaces |
| **08 - Research programme** | Four open problems |
| **09 - About & References** | Research context, bibliography |

---

## Technical notes

- **Zero dependencies** except Google Fonts and MathJax (both CDN, no install needed)
- All diagrams and animations are pure HTML5 Canvas + JS
- MathJax renders all mathematical notation
- Fully responsive (mobile/tablet/desktop)
- No build step required: `index.html` is the entire site

---

## Running locally

```bash
git clone https://github.com/khadiijatu/morse-geometry-ct.git
cd morse-geometry-ct
# Open index.html in any modern browser
```

---

## Author

**K. Cissé**
