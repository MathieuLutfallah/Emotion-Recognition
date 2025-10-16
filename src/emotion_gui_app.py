from .config import Config
from . import gui
from src import networks
from src import utils
from src import faceAligner
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Tuple, List
from src import pickROI
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
import pyautogui
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import dlib
print("Dlib compiled with CUDA:", dlib.DLIB_USE_CUDA)
print("CUDA devices available:", dlib.cuda.get_num_devices())

class ROI:
    def __init__(self, center_x, center_y, width, height):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
    def region(self) -> Tuple[int, int, int, int]:
        x = int(self.center_x - self.width // 2)
        y = int(self.center_y - self.height // 2)
        return (x, y, self.width, self.height)




def run(parameters: str = "parameters.yaml", frames_per_batch: int = 3, iterations: int = 100):
    cfg = Config(parameters)
    #plt.ion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    model = networks.Network(cfg.network)
    model.load_state_dict(torch.load(cfg.loadnet_path))
    model.cuda()
    model.eval()
    transformValidation=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])


    fa = faceAligner.FaceAligner(shape_predictor_path=cfg.predictor_path,
                 cnn_detector_path=cfg.cnn_face_path,
                 chip_size=224,
                 padding=0.25)

    # Load assets
    assets_dir = Path(cfg.paths.get("assets_dir", "assets"))
    images = {
    "happy": Image.open(assets_dir / "happy.jpg"),
    "neutral": Image.open(assets_dir / "neutral.jpg"),
    "sad": Image.open(assets_dir / "sad.jpg"),
    }

    roi = pickROI.pick_roi()
    


    for i in range(iterations):
        frames: List[torch.Tensor] = []

        # Collect a small batch of screenshots
        print("collecting batch")
        for _ in range(frames_per_batch):
            img = pyautogui.screenshot(region=roi) # RGB
            frame = np.array(img)

            print("searching for faces")
            aligned_bgr = fa.align(frame)
            if aligned_bgr is None:
                print("no face found")
                gui.show_no_face(frame)
                print("Face not found; skipping frame")
                continue

            print("face found")

            frame = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

            frame=transformValidation(Image.fromarray(frame))
            frames.append(frame)


        if not frames:
            continue


        with torch.no_grad():
            batch = torch.stack(frames, dim=3).unsqueeze(0).to(device)
            print(batch.size())
            scores = model(batch)
            print(scores)
            probs = F.softmax(scores, dim=0).detach().cpu().numpy()
            emotion_idx = int(np.argmax(probs))


        gui.show_emotion(probs, emotion_idx, images)
        print(f"Iter {i} → emotion={emotion_idx}, probs={probs}")
