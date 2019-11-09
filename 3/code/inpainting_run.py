import os
import inpainting
import cv2


in_img = cv2.imread('../input/tulips_in.png')
mask_img = cv2.imread('../input/tulips_mask.png')
mask = mask_img[:, :, 0].astype(bool, copy=False)
out_img = in_img.copy()

inpainting.inpaint(out_img, mask, 3)
cv2.imwrite('../output/inpainting/output.png', out_img)
