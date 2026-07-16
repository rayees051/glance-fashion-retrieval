import os
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import open_clip
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

class FashionIndexer:

    def __init__(self,
                 image_dir,
                 output_dir,
                 device="cuda"):

        self.image_dir = image_dir
        self.output_dir = output_dir
        self.device = device