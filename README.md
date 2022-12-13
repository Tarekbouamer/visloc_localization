# visloc_localization

- [x]  remove Models folder and replace with retrieval packages and other features in thirdparty.
- [x]  add superpoints and superglue.
- [ ]  add cameras to dataset
- [ ]  separate the modules Mappers, Retrieval, Matcher, and Localizer
- [ ]  make a nice visualization.

## Results:

| Methods                | Aachen day         | Aachen night       | Retrieval                    |
| ---------------------- | ------------------ | ------------------ | ---------------------------- |
| SuperPoint + NN        | 89.6 / 95.4 / 98.8 | 86.7 / 93.9 / 100  | sfm_resnet50_gem_2048 (k=50) |
| SuperPoint + SuperGlue | 89.6 / 95.4 / 98.8 | 86.7 / 93.9 / 100  | NetVLAD top 50               |
