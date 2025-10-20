# Emotion Recognition — A Tool to Improve Meeting Experience for Visually Impaired

This repository contains the implementation and experiments from my **Master’s Thesis**, which also served as the foundation for the publication:

> **Mathieu Lutfallah, Benno Käch, Christian Hirt, and Andreas Kunz**  
> *Emotion Recognition – A Tool to Improve Meeting Experience for Visually Impaired*  
> In: *Computers Helping People with Special Needs*, pp. 305–312, Springer, Cham (2022).  
> ISBN: 978-3-031-08648-9  
> [DOI: 10.1007/978-3-031-08648-9_35](https://doi.org/10.1007/978-3-031-08648-9_35)

---

## 🧠 Abstract

Facial expressions play an important role in human communication since they enrich spoken information and help convey additional sentiments such as mood. Among others, they non-verbally express a partner’s agreement or disagreement to spoken information. Together with the audio signal, humans can even detect nuances of mood changes.  

However, facial expressions remain inaccessible to blind and visually impaired individuals, and the voice signal alone might not carry enough mood information.  
This work presents a **real-time emotion recognition system** that detects a user’s facial expressions and communicates their emotional state to assistive interfaces.

---

## ⚙️ Features

- **Real-time emotion detection** from selected on-screen regions.  
- User selects an area on the screen — the system takes a screenshot and detects the face within.  
- Outputs **probabilities for three emotions**:  
  - 😐 *Neutral*  
  - 😊 *Happy*  
  - 😢 *Sad*  
- Visual feedback using emojis.  
- Configurable deep network structure and weights defined in `parameters.yaml`.  
- Face detection and alignment using **dlib** (GPU-accelerated version recommended).  
- All files and scripts tested on **Ubuntu Linux**.

## 🧠 Training (Coming Soon)

The repository will later include training scripts for:

CK+ dataset (Cohn–Kanade Extended)

EmotiW dataset (Emotion Recognition in the Wild)

These scripts will demonstrate how to retrain or fine-tune the model architecture defined in parameters.yaml using your own data.

## 📜 Citation
If you use this repository in your research, please cite:
@InProceedings{10.1007/978-3-031-08648-9_35,
  author    = {Lutfallah, Mathieu and K{\"a}ch, Benno and Hirt, Christian and Kunz, Andreas},
  editor    = {Miesenberger, Klaus and Kouroupetroglou, Georgios and Mavrou, Katerina and Manduchi, Roberto and Covarrubias Rodriguez, Mario and Pen{\'a}z, Petr},
  title     = {Emotion Recognition - A Tool to Improve Meeting Experience for Visually Impaired},
  booktitle = {Computers Helping People with Special Needs},
  year      = {2022},
  publisher = {Springer International Publishing},
  address   = {Cham},
  pages     = {305--312},
  isbn      = {978-3-031-08648-9}
}


