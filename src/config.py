from pathlib import Path
from ruamel.yaml import YAML


class Config:
	def __init__(self, path: str = "parameters.yaml"):
		yaml = YAML(typ="safe")
		with open(path, "r", encoding="utf-8") as f:
			self.raw = yaml.load(f)


		self.loadnet_path = Path(self.raw["loadnet_path"]).resolve()
		self.network = self.raw.get("networkRT", {})
		self.paths = self.raw.get("paths", {})

		self.roi = self.raw.get("roi", {})

		# Validate expected files
		self.predictor_path = Path(self.paths["dlib_predictor"]).resolve()
		self.cnn_face_path = Path(self.paths["dlib_cnn"]).resolve()
		for p in [self.loadnet_path, self.predictor_path, self.cnn_face_path]:
			if not p.exists():
				raise FileNotFoundError(f"Missing required file: {p}")