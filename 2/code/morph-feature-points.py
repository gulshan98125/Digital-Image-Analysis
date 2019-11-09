import numpy as np
import cv2

DIR = "../data/"

def zoom(img,k):
	# Repeat cols
	zoomed = np.repeat(img,k,axis=1)
	# Repeat rows
	zoomed = np.repeat(zoomed,k,axis=0)
	return zoomed

ZOOM = 3
filenames = ['arnie.png', 'bush.png']
images = []
images.append(zoom(cv2.imread(DIR + filenames[0], 1), ZOOM))
images.append(zoom(cv2.imread(DIR + filenames[1], 1), ZOOM))

file = []
file.append(open(DIR + filenames[0] + "-morph-points.txt","w"))
file.append(open(DIR + filenames[1] + "-morph-points.txt","w"))

def addBoundaryPoints(img, fileObj, ZOOM):
	height,width = (img.shape[0]/ZOOM)-1,(img.shape[1]/ZOOM)-1
	fileObj.write("%d %d\n"%(0,0))
	fileObj.write("%d %d\n"%(0,height))
	fileObj.write("%d %d\n"%(0,height/2))
	fileObj.write("%d %d\n"%(width, height))
	fileObj.write("%d %d\n"%(width, height/2))
	fileObj.write("%d %d\n"%(width, 0))

addBoundaryPoints(images[0],file[0],ZOOM)
addBoundaryPoints(images[1],file[1],ZOOM)


def saveXY(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONUP:
		file[param-1].write(str(x/ZOOM) + " " + str(y/ZOOM) + "\n")
		cv2.circle(images[param-1],(x,y), 2*ZOOM, (0,255,0), -1)
		cv2.imshow('image'+str(param),images[param-1])
		# img2 = cv2.imread('../data/' + filenames[param-1], 1)
		# cv2.circle(img2,(x/ZOOM,y/ZOOM), 2, (0,255,0), -1)
		# cv2.imshow('image',img2)


cv2.imshow('image1',images[0])
cv2.imshow('image2',images[1])
cv2.setMouseCallback("image1", saveXY, 1)
cv2.setMouseCallback("image2", saveXY, 2)


while(True):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()