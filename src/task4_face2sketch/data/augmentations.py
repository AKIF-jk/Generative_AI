import random
from torchvision import transforms
from torchvision.transforms import functional as F

class PairedTransform:
    def __init__(self, size=128):
        self.size = size
        # ToTensor converts PIL Image to tensor [0.0, 1.0]
        # Normalize maps to [-1.0, 1.0] which is standard for GANs with Tanh output
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def __call__(self, photo, sketch):
        # Resize both to 128x128
        photo = F.resize(photo, [self.size, self.size])
        sketch = F.resize(sketch, [self.size, self.size])
        
        # Random horizontal flip (shared state)
        if random.random() > 0.5:
            photo = F.hflip(photo)
            sketch = F.hflip(sketch)
            
        # Convert to tensor
        photo = F.to_tensor(photo)
        sketch = F.to_tensor(sketch)
        
        # Normalize
        photo = self.normalize(photo)
        sketch = self.normalize(sketch)
        
        return photo, sketch

class PairedValTransform:
    def __init__(self, size=128):
        self.size = size
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def __call__(self, photo, sketch):
        # Resize both to 128x128
        photo = F.resize(photo, [self.size, self.size])
        sketch = F.resize(sketch, [self.size, self.size])
        
        # Convert to tensor
        photo = F.to_tensor(photo)
        sketch = F.to_tensor(sketch)
        
        # Normalize
        photo = self.normalize(photo)
        sketch = self.normalize(sketch)
        
        return photo, sketch
