import PIL.Image as im
import numpy as np

img = im.open('out_us.jpg')
img2 = im.open('out9.jpg')
# img.show()
img_array = np.array(img) / (1.0 * 255)
print img_array
img2_array = np.array(img2) / 255.0
print img2_array
dist = np.linalg.norm(img_array - img2_array)
print dist
