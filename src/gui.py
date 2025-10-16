from typing import Sequence
import matplotlib
import matplotlib.pyplot as plt


# Use TkAgg only if available
try:
    matplotlib.use("TkAgg")
except Exception:
    pass




def show_emotion(probabilities, emotion_idx, images, _cache={}):
    """
    Update a persistent figure instead of creating a new one each call.
    """
    # create once
    if "fig" not in _cache:
        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        # init bars with zeros; keep the BarContainer
        bars = ax[0].bar(["+", "neutral", "-"], [0, 0, 0], align="center")
        ax[0].set_xlabel("Emotions")
        ax[0].set_ylabel("Predicted score")
        ax[0].set_ylim(0, 1)
        ax[0].set_title("Emotion Predicted")

        # init image with neutral (anything is fine)
        im = ax[1].imshow(images["neutral"])
        ax[1].set_title("neutral")
        ax[1].axis("off")

        plt.tight_layout()
        _cache.update(dict(fig=fig, ax=ax, bars=bars, im=im))

    fig   = _cache["fig"]
    ax    = _cache["ax"]
    bars  = _cache["bars"]
    im    = _cache["im"]

    # update bar heights
    for rect, p in zip(bars, probabilities):
        rect.set_height(float(p))

    # choose image/title (match your indices)
    if emotion_idx == 2:
        im.set_data(images["sad"]);      title = "-"
    elif emotion_idx == 1:
        im.set_data(images["neutral"]);  title = "neutral"
    else:
        im.set_data(images["happy"]);    title = "+"

    ax[1].set_title(title)

    # redraw efficiently
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.5)  # tiny yield to UI loo




def show_no_face(frame_rgb):
    fig, ax = plt.subplots(1, 2, figsize=(7, 4))

    # Left: message
    ax[0].axis("off")
    ax[0].text(
        0.5, 0.5, "No face detected",
        ha="center", va="center", fontsize=16, fontweight="bold"
    )

    # Right: the current frame (or ROI)
    ax[1].imshow(frame_rgb)
    ax[1].set_title("Current view")
    ax[1].axis("off")

    plt.tight_layout()
    plt.pause(2)
    plt.close(fig)   # close the figure after showing
