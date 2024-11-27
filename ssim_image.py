import cv2 
from dataclasses import dataclass
import numpy.typing as npt


@dataclass
class Images:
    Depth: npt.NDArray
    Pred: npt.NDArray

class SSIM_Loss():
    def __init__(self, depth_img, pred_img):
        self.images = self.read_images(depth_img, pred_img)
        k_1, k_2, L = 0.01, 0.03, 255
        self.constant_1, self.constant_2, self.constant_3 = (k_1 * L) ** 2, (k_2 * L) ** 2, (k_2 * L) ** 2 / 2
        

    def read_images(self, depth, pred) -> Images:
        try: 
            img_depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)
            img_pred = cv2.imread(pred, cv2.IMREAD_GRAYSCALE)

            return Images(Depth=img_depth, img_pred=img_pred)
        except Exception as e:
            print(e)

    def window(self, window_size=11, stride=1):
        images = self.images
        h, w, ws = images.Depth.shape, window_size

        for y in range(0, h - ws + 1, stride):
            for x in range(0, w - ws + 1, stride):
                patch = images.Depth[y: y + ws, x:x + ws]
                yield patch, (y, x)