![demo_vid](assets/VisLoc-logos.jpeg)

# Visloc-Localization: a Visual Localization Library based.

## Content
- [Introduction](#introduction)
- [Welcome](#welcome)
- [Roadmap](#roadmap)
- [Results](#results)
- [Licenses](#licenses)
- [Citing](#citing)

## Introduction:

Visloc-Localization library is a collection of various localization algorithmsn and pose estimation models for robotics and autonomous system applications


## Welcome

* Welcome to the `Visloc-Localizatio` :sparkles:


## Roadmap

- [x]  remove Models folder and replace with retrieval packages and other features in thirdparty.
- [x]  add superpoints and superglue.
- [ ]  add cameras to dataset
- [ ]  separate the modules Mappers, Retrieval, Matcher, and Localizer
- [ ]  make a nice visualization.

## Results:

| Methods                | Aachen day         | Aachen night       | Retrieval                    |
| ---------------------- | ------------------ | ------------------ | ---------------------------- |
| SP_NN_PyColmap         | 85.1 / 93.0 / 96.4 | 68.4 / 85.7 / 94.9 | sfm_resnet50_gem_2048 (k=50) |
| SP_NN_Covis_PyColmap   | 85.4 / 93.0 / 96.6 | 68.4 / 86.7 / 94.9 | sfm_resnet50_gem_2048 (k=50) |
| SP_NN_Covis_PoseLib    | 86.5 / 93.6 / 96.8 | 72.4 / 83.7 / 93.9 | sfm_resnet50_gem_2048 (k=50) |


run again SuperPoint + NN for same image pairs as SuperPoint + NN + covis
