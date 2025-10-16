import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
class FaceAligner:
    """
    End-to-end face detection + alignment with fallback preprocessing.
    - Primary: HOG detector
    - Fallbacks: CNN detector, then Linear, CLAHE, HistEq
    - Output: aligned face chip (BGR, 224x224) or None
    """
    def __init__(
        self,
        shape_predictor_path: str,
        cnn_detector_path: str = None,
        chip_size: int = 224,
        padding: float = 0.25
    ):
        # Detectors / predictors
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(str(shape_predictor_path))

        self.cnn_face_detector = (
            dlib.cnn_face_detection_model_v1(str(cnn_detector_path))
            if cnn_detector_path else None
        )

        # Alignment params
        self.chip_size = chip_size
        self.padding = padding

    # --------------------------- public API ---------------------------

    def align(self, bgr_image: np.ndarray):
        """
        Takes a BGR image, returns aligned BGR face chip or None if not found.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        # dlib expects RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        original = rgb.copy()

        # 1) HOG
        dets = self.detector(rgb, 1)

        # 2) CNN + preprocessing fallbacks
        if len(dets) == 0:
            dets = self._cnn_detect(rgb)
        if len(dets) == 0:
            dets = self._cnn_detect(self._linear_equal(rgb))
        if len(dets) == 0:
            dets = self._cnn_detect(self._clahe_color(rgb))
        if len(dets) == 0:
            dets = self._cnn_detect(self._hist_eq_color(rgb))
        if len(dets) == 0:
            return None

        # Landmarks -> aligned chip
        faces = dlib.full_object_detections()
        for d in dets:
            faces.append(self.sp(original, d))
        chip = dlib.get_face_chip(original, faces[0], size=self.chip_size, padding=self.padding)

        # back to BGR for OpenCV callers
        return cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)

    # ------------------------ private helpers ------------------------

    def _cnn_detect(self, rgb: np.ndarray) -> dlib.rectangles:
        rects = dlib.rectangles()
        if self.cnn_face_detector is None:
            return rects
        cnn_dets = self.cnn_face_detector(rgb, 1)
        rects.extend([d.rect for d in cnn_dets])
        return rects

    def _clahe_color(self, rgb: np.ndarray) -> np.ndarray:
        # Work in YCrCb; apply CLAHE on Y only, then convert back to RGB
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        y = clahe.apply(y)
        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    def _hist_eq_color(self, rgb: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    def _linear_equal(self, rgb: np.ndarray) -> np.ndarray:
        # Stretch on luminance channel
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        hist = cv2.calcHist([y], [0], None, [256], [0, 256]).ravel()
        # find first/last nonzero bins
        nz = np.flatnonzero(hist)
        if len(nz) == 0:
            return rgb
        min_bin, max_bin = int(nz[0]), int(nz[-1])

        # build LUT
        lut = np.zeros(256, dtype=np.uint8)
        if max_bin > min_bin:
            lut[min_bin:max_bin + 1] = np.clip(
                (np.linspace(0, 255, max_bin - min_bin + 1) + 0.5).astype(np.int32), 0, 255
            )
        y = cv2.LUT(y, lut.astype(np.uint8))

        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)