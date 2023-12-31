import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

path_dir = r'C:\projects\VideoSketch\results\debug'

im_name = '1'
uv = np.load(str(Path(path_dir) / 'originals' / 'flow12-1.npy'))
us = uv[:, :, 0] # cv2.imread(str(Path(path_dir) / f'{im_name}_u.png'))[:, :, 0]
vs = uv[:, :, 1] # cv2.imread(str(Path(path_dir) / f'{im_name}_v.png'))[:, :, 0]
orig = cv2.imread(str(Path(path_dir) / f'{im_name}_orig.png'))
center = cv2.imread(str(Path(path_dir) / '12_orig.png'))
print(uv.shape)
h, w = us.shape[:2]

us = us.astype(int)
#us = us * (us < w//2) + (w - us) * (us >= w//2)
vs = vs.astype(int)
#vs = vs * (vs < h//2) + (h - vs) * (vs >= h//2)

#us -= h//2
#vs -= w//2



new_image = np.zeros_like(orig, dtype=np.uint8)
new_image_vu = np.zeros_like(orig, dtype=np.uint8)
new_image_minus = np.zeros_like(orig, dtype=np.uint8)
new_image_minus_vu = np.zeros_like(orig, dtype=np.uint8)
for y in range(h):
    for x in range(w):
        u, v = us[y, x], vs[y, x]
        new_image[y, x, :] = center[max(0, min(h-1, (y+u))),
                                    max(0, min(w-1, (x+v))), :]
        
        new_image_vu[y, x, :] = center[max(0, min(h-1, (y+v))),
                                       max(0, min(w-1, (x+u))), :]
        
        new_image_minus[y, x, :] = center[max(0, min(h-1, (y-u))),
                                          max(0, min(w-1, (x-v))), :]
        
        new_image_minus_vu[y, x, :] = center[max(0, min(h-1, (y-v))),
                                             max(0, min(w-1, (x-u))), :]
fig, ax = plt.subplots(3, 4)
ax[0, 1].imshow(center)
ax[0, 1].set_title('center')
ax[0, 2].imshow(orig)
ax[0, 2].set_title('orig')
ax[1, 0].imshow(np.abs(new_image - orig))
ax[1, 0].set_title('new_image')
ax[1, 1].imshow(np.abs(new_image_vu - orig))
ax[1, 1].set_title('vu')
ax[1, 2].imshow(np.abs(new_image_minus - orig))
ax[1, 2].set_title('minus')
ax[1, 3].imshow(np.abs(new_image_minus_vu - orig))
ax[1, 3].set_title('minus_vu')
ax[2, 0].imshow(new_image)
ax[2, 0].set_title('new_image')
ax[2, 1].imshow(new_image_vu)
ax[2, 1].set_title('vu')
ax[2, 2].imshow(new_image_minus)
ax[2, 2].set_title('minus')
ax[2, 3].imshow(new_image_minus_vu)
ax[2, 3].set_title('minus_vu')
ax[0, 0].imshow(us)
ax[0, 0].set_title('u')
ax[0, 3].imshow(vs)
ax[0, 3].set_title('v')
plt.show()
#plt.imshow(us - w//2)
#plt.show()
#plt.imshow(vs - h//2)
#plt.show()