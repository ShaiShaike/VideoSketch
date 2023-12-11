import numpy as np
import cv2
from PIL import Image

class Mosaic:
    def __init__(self, imshape, padding=10) -> None:
        self.padding = padding
        self.h_mosaic = 2 * imshape[0] + 3 * padding
        self.mosaic = np.uint8(255 * np.ones((self.h_mosaic, padding, imshape[2])))

    def add(self, image, orig_image):
        print('shapes min max:', image.shape, orig_image.shape, np.min(image), np.max(image), np.min(orig_image), np.max(orig_image), self.mosaic.shape)
        h_padding = np.uint8(255 * np.ones((self.h_mosaic, self.padding, self.mosaic.shape[2])))
        v_padding = np.uint8(255 * np.ones((self.padding, image.shape[1], self.mosaic.shape[2])))
        self.mosaic = cv2.hconcat([self.mosaic,
                                   h_padding,
                                   cv2.vconcat([v_padding,
                                                cv2.resize(orig_image, image.shape[:2]),
                                                v_padding,
                                                image,
                                                v_padding])
                                    ])
    def save(self, path):
        print('save mosaic 1')
        h_padding = np.uint8(255 * np.ones((self.h_mosaic, self.padding, self.mosaic.shape[2])))
        print('save mosaic 2')
        self.mosaic = cv2.hconcat([self.mosaic, h_padding])
        print('save mosaic 3')
        print('mosaic shape:', self.mosaic.shape)
        print(str(path))
        cv2.imwrite(str(path), self.mosaic)
        cv2.imwrite(f"/content/gdrive/My Drive/Final Project_206899080/results/debug/{str(path).split('//')[-1]}")
        Image.fromarray(self.mosaic).save(f"/content/gdrive/My Drive/Final Project_206899080/results/debug/PIL_{str(path).split('//')[-1]}")