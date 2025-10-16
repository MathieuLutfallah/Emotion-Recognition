import pyautogui
import cv2
import numpy as np

def pick_roi(initial_region=None):
    """
    Lets the user drag-select a region on a screenshot and confirm it.
    Returns (x, y, w, h).
    """
    while True:
        # Take a full screenshot for selection
        full = pyautogui.screenshot()
        full_np = cv2.cvtColor(np.array(full), cv2.COLOR_RGB2BGR)

        # Optional visual hint for previous selection
        if initial_region is not None:
            x, y, w, h = initial_region
            cv2.rectangle(full_np, (x, y), (x+w, y+h), (0, 255, 255), 2)

        win = "Select capture region (Enter=OK, c/Esc=Cancel)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        x, y, w, h = cv2.selectROI(win, full_np, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)

        if w == 0 or h == 0:
            print("Selection canceled.")
            if initial_region is not None:
                return initial_region
            else:
                continue  # re-ask
        region = (int(x), int(y), int(w), int(h))
        preview_roi(region,full_np)
        cv2.waitKey(1)

        # --- NEW: Ask user if they’re happy with it ---
        resp = input(f"Selected ROI: {region}. Keep it? [y/n]: ").strip().lower()
        if resp in ("y", "yes", ""):
            return region
        else:
            print("Re-selecting ROI...")
            initial_region = region  # use last as visual reference
            continue


def preview_roi(region, screenshot):
    """
    Shows the selected ROI directly on the provided screenshot.
    Lets the user confirm visually if the selection is correct.

    Args:
        region (tuple): (x, y, w, h)
        screenshot (numpy.ndarray): BGR image (e.g. from cv2 or pyautogui)
    """
    x, y, w, h = region
    preview = screenshot.copy()

    # Draw ROI rectangle
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        preview,
        f"ROI: {x},{y},{w},{h}",
        (x + 10, y - 10 if y > 20 else y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ROI Preview", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("ROI Preview", preview)
    print(f"ROI: {x},{y},{w},{h}")
    print("Press 'y' to confirm, 'n' to redo, or 'Esc' to exit.")

    cv2.waitKey(0)
    cv2.destroyWindow("ROI Preview")
    return


# Example usage
if __name__ == "__main__":
    region = pick_roi()
    print("Final ROI:", region)

