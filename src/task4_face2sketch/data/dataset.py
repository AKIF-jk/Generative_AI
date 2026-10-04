import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class FS2KPairedDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        anno_file = os.path.join(data_dir, f'anno_{split}.json')
        with open(anno_file, 'r') as f:
            all_annotations = json.load(f)

        # Filter out entries with missing files
        self.annotations = [a for a in all_annotations if self._files_exist(a)]
        skipped = len(all_annotations) - len(self.annotations)
        if skipped:
            print(f"[FS2KPairedDataset] Skipped {skipped} samples with missing files in '{split}' split.")
            
        self.style_map = {0: 0, 1: 1, 2: 2}

    def __len__(self):
        return len(self.annotations)

    def _files_exist(self, anno):
        """Check if both photo and sketch files exist for an annotation."""
        image_name = anno['image_name']
        parts = image_name.split('/')
        sketch_folder = parts[0].replace('photo', 'sketch')
        sketch_file = parts[1].replace('image', 'sketch')
        try:
            self._resolve_path(os.path.join(self.data_dir, 'photo', image_name))
            self._resolve_path(os.path.join(self.data_dir, 'sketch', sketch_folder, sketch_file))
            return True
        except FileNotFoundError:
            return False

    @staticmethod
    def _resolve_path(path_without_ext):
        """Find the actual file by trying common image extensions."""
        for ext in ('.jpg', '.png', '.jpeg'):
            full = path_without_ext + ext
            if os.path.isfile(full):
                return full
        raise FileNotFoundError(
            f"No image found for: {path_without_ext} (tried .jpg, .png, .jpeg)"
        )

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        
        # image_name is e.g. "photo1/image0110" — files live at photo/photo1/image0110.jpg
        image_name = anno['image_name']
        photo_path = self._resolve_path(
            os.path.join(self.data_dir, 'photo', image_name)
        )
        
        # Derive sketch path: photo1 → sketch1, image → sketch
        parts = image_name.split('/')  # ['photo1', 'image0110']
        sketch_folder = parts[0].replace('photo', 'sketch')  # 'sketch1'
        sketch_file = parts[1].replace('image', 'sketch')     # 'sketch0110'
        sketch_path = self._resolve_path(
            os.path.join(self.data_dir, 'sketch', sketch_folder, sketch_file)
        )
        
        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        
        style = self.style_map[anno['style']]
        
        if self.transform:
            photo, sketch = self.transform(photo, sketch)
            
        return photo, sketch, style

