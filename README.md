# Generative_AI

## Shared Oxford-IIIT Pet pipeline (Tasks 1--3)

The pipeline uses official `annotations/trainval.txt` and `annotations/test.txt`
as the source of truth, ignores breed labels, converts every image to RGB, and
resizes it to 128×128. It does not use a filename glob for the split.

Install the data-pipeline dependencies and create the artefacts once:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 scripts/prepare_pet_data.py --download
```

If `data/images` and `data/annotations` already exist, omit `--download`. The
script writes these protected, reusable artefacts:

- `data/manifests/development_split.csv` — 2,944 train and 736 validation
  official development images, shuffled with seed 42.
- `data/manifests/validation_manifest.json` — clean plus low/medium/high rows
  for each corruption on every validation image.
- `data/manifests/test_manifest.json` — the same ten rows per untouched
  official test image. The three fixed severities exactly follow the brief.

The script refuses to replace existing artefacts. Use `--force` only when an
intentional regeneration is required.

```python
from pet_restoration import PetRestorationDataset

train = PetRestorationDataset("data", "train", split_csv="data/manifests/development_split.csv")
validation = PetRestorationDataset("data", "validation", manifest_path="data/manifests/validation_manifest.json")
test = PetRestorationDataset("data", "test", manifest_path="data/manifests/test_manifest.json")
corrupted, clean, corruption_label = train[0]
```

`train` samples clean, salt-and-pepper, Gaussian blur, and occlusion uniformly
on every `__getitem__`. Validation/test replay their manifest records exactly.
Labels are `clean=0`, `salt_pepper=1`, `gaussian_blur=2`, `occlusion=3`.

Run automated and visual checks before training:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/visualize_pet_corruptions.py
```
