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


## Task 4: Face-to-Sketch Generator (Conditional GAN)

Task 4 uses the FS2K dataset and trains a Conditional GAN to generate sketches from faces.

### Reproducing Task 4 on Google Colab (Recommended)
Since GAN training is computationally expensive, we recommend running the provided Jupyter Notebook in Google Colab.
1. Download or clone this repository to your Colab environment.
2. Upload the `FS2K` dataset to `data/FS2K/` in your Colab environment.
3. Open `notebooks/task4_colab_workflow.ipynb` in Colab.
4. Run all cells to perform data auditing, Optuna HPO, full training, evaluation, and ONNX export.
5. Download the final `generator.onnx` file from Colab and place it in the `onnx_models/` folder on your local machine.

### Reproducing Task 4 Locally
If you have a local GPU, run the following commands sequentially:

```bash
# 1. Ensure dataset is at data/FS2K
# 2. Run Hyperparameter Search (Optuna)
PYTHONPATH=src/task4_face2sketch python3 src/task4_face2sketch/optuna_search.py

# 3. Run Full Training (Update task4.yaml with best params if desired)
PYTHONPATH=src/task4_face2sketch python3 src/task4_face2sketch/train.py --config src/task4_face2sketch/configs/task4.yaml

# 4. Evaluation
PYTHONPATH=src/task4_face2sketch python3 src/task4_face2sketch/evaluate.py

# 5. Export to ONNX
PYTHONPATH=src/task4_face2sketch python3 src/task4_face2sketch/export_onnx.py
```

### Running the Application Workspace
Once you have the `generator.onnx` file, place it in `onnx_models/generator.onnx`.
Start the backend via Docker Compose:

```bash
docker-compose up --build
```
The backend API is accessible at `http://localhost:8000`. You can integrate the provided React Component located at `frontend/src/workspaces/FaceToSketchGenerator/index.jsx` into your frontend application!
