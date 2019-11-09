import numpy as np
import cv2

# assume greyscale
# zoomByReplication
def zoom(x,y,img):
	HEIGHT,WIDTH = img.shape
	part = img[max(0,y-100):min(y+100,HEIGHT),max(0,x-100):min(x+100,WIDTH)]

	ZOOM = 3

	# Repeat cols
	zoomed = np.repeat(part,ZOOM,axis=1)
	# Repeat rows
	zoomed = np.repeat(zoomed,ZOOM,axis=0)
	return zoomed

def click_and_crop(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		cv2.imshow('frame2',zoom(x,y,img2))

image = ['car.jpg', 'image2.jpg']
img = cv2.imread('data/' + image[0], 1)
img2 = cv2.imread('data/' + image[0], 0)

cv2.imshow('frame1',img2)
cv2.setMouseCallback("frame1", click_and_crop)


while(True):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()