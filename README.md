# visloc_localization

- [x]  remove Models folder and replace with retrieval packages and other features in thirdparty.
- [x]  add superpoints and superglue.
- [ ]  add cameras to dataset
- [ ]  separate the modules Mappers, Retrieval, Matcher, and Localizer
- [ ]  make a nice visualization.

## Results:

| Methods                | Aachen day         | Aachen night       | Retrieval                    |
| ---------------------- | ------------------ | ------------------ | ---------------------------- |
| SuperPoint + NN        | 85.1 / 93.0 / 96.4 | 68.4 / 85.7 / 94.9 | sfm_resnet50_gem_2048 (k=50) |
| SuperPoint + NN + covis| 85.4 / 93.0 / 96.6 | 68.4 / 86.7 / 94.9 | sfm_resnet50_gem_2048 (k=50) |


run again SuperPoint + NN for same image pairs as SuperPoint + NN + covis
