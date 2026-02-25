# SydneyMTL
Repository for SydneyMTL: Interpretable Multi-Task Learning for Complete Sydney System Assessment in Gastric Biopsies

## 🧠 Overview
<p align="center">
  <img src="docs/images/Figure_0_Graphical_Abstract.png" width="40%">
</p>

## 🔬 Key Idea
- **Unified multi-task learning for the Updated Sydney System (USS)**: A single weakly supervised MIL framework predicts all five USS attributes simultaneously: H. pylori, Intestinal Metaplasia, Glandular Atrophy, Neutrophil Activity, and Mononuclear Cell Infiltration, following the 4-tier severity grading (0–3). Atrophy additionally includes an explicit N/A class to reflect real-world diagnostic workflow.
- **Task-specific attention for interpretability**: The model uses shared slide representations with task-specific attention pooling and classification heads, enabling attribute-wise heatmaps that highlight regions contributing to each grading decision.
- **Long-tail aware optimization via prior-based logit adjustment**: To address severe class imbalance (dominance of Absent/Mild cases), we incorporate empirical class priors directly into logits. This improves robustness on rare Moderate/Marked grades and maintains performance under balanced evaluation.
- **Emergent ordinal structure in representation space**: Although trained with standard classification loss, the learned embeddings preserve biological ordinality—severity grades form a continuous trajectory in latent space.

## 📊 Results
- **Large-scale validation**: Evaluated on 50,765 retrospective WSIs and a 366-case expert-consensus “Golden” dataset with balanced severity distribution.
- **Robust performance under balanced evaluation**: While baseline methods degrade substantially on the Golden set, our model maintains high agreement (e.g., QWK up to 0.898 for IM and 0.826 for H. pylori).
- **Clinically meaningful agreement with pathologists**: Across 24 pathologists, the model achieves strong concordance (mean QWK ≈ 0.73), reflecting real-world diagnostic consistency.
- **AI-assisted reading improves consistency and efficiency**: In a randomized reader study, AI support increased inter-observer agreement and reduced reading time by ~34% per WSI.
- **Pathologically plausible explanations**: Attention maps localize biologically meaningful structures (e.g., goblet cells for IM, neutrophil infiltration for activity), supporting clinical interpretability.

## 👨‍🔬 Authors

| Name              | ORCID                            | Email                               | Affiliation                                   | Notes                 |
|-------------------|----------------------------------|-------------------------------------|-----------------------------------------------|------------------------|
| **Ho Heon Kim**   | [0000-0001-7260-7504](https://orcid.org/0000-0001-7260-7504) | hoheon0509@mf.seegene.com          | $^{1}$ AI Research Center, Seegene Medical Foundation | *Contributed equally* |
| **Won Chang Jeong** | [0009-0008-1931-5957](https://orcid.org/0009-0008-1931-5957) | jeongwonchan53@gmail.com      | $^{1}$ AI Research Center, Seegene Medical Foundation | *Contributed equally*|
| **Yuri Hwang**    | - | -             | $^{1}$ AI Research Center, Seegene Medical Foundation |
| **Gisu Hwang**    | [0000-0003-1046-9286](https://orcid.org/0000-0003-1046-9286) | gshwang@mf.seegene.com             | $^{1}$ AI Research Center, Seegene Medical Foundation |
| **Kyungeun Kim**   | - | kekim@mf.seegene.com             | $^{1,2}$ AI Research Center / Pathology Center, Seegene Medical Foundation | *Corresponding author* |
| **Young Sin Ko**   | [0000-0003-1319-4847](https://orcid.org/0000-0003-1319-4847) | noteasy@mf.seegene.com             | $^{1,2}$ AI Research Center / Pathology Center, Seegene Medical Foundation | *Corresponding author* |

### 📍 Affiliations
- $^{1}$ AI Research Center, Seegene Medical Foundation, 288 Dapsimni-ro, Seoul, South Korea  
- $^{2}$ Pathology Center, Seegene Medical Foundation, 288 Dapsimni-ro, Seoul, South Korea