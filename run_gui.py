import argparse
from src.emotion_gui_app import run


if __name__ == "__main__":
	p = argparse.ArgumentParser()
	p.add_argument("--params", default="parameters.yaml")
	args = p.parse_args()
	run(parameters=args.params)