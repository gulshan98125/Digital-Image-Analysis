import cv2
import numpy as np
import imutils

def rotateImage(img, angle):
	rows,cols = img.shape[:2]
	dst = imutils.rotate_bound(img, angle)
	return dst

def angleWithHorizontal(p1,p2):
	p1,p2 = sorted([p1,p2])
	x1,y1 = p1
	x2,y2 = p2
	inRad = (np.arctan((y2-y1)/(x2-x1)))
	inDeg = 180/np.pi*inRad
	return inDeg

def getDistance(p1,p2):
	return np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)

def getResizingFactorSunglass(filterPoints, eyePoints):
	d1 = getDistance(filterPoints[0], filterPoints[1])
	d2 = getDistance(eyePoints[0], eyePoints[1])
	return d2/d1

def getRectangle(faceImg, glassImg, eyePoints, filterPoints, factor):
	H,W = glassImg.shape[:2]
	eyePoints = sorted(eyePoints)
	filterPoints = sorted(filterPoints)
	xl,yl = filterPoints[0]
	yl = H-yl
	xl,yl = xl*factor, yl*factor
	bottomLeft = (eyePoints[0][0]-xl, eyePoints[0][1]+yl)
	xr,yr = filterPoints[1]
	xr = W-xr
	xr,yr = xr*factor, yr*factor
	topRight = (eyePoints[1][0]+xr, eyePoints[1][1]-yr)
	return [bottomLeft,topRight]

def handleResizedFilter(resized,rect,faceImg):
	x1,y1 = rect[0]
	x2,y2 = rect[1]
	ox1,oy1 = x1,y2
	rows,cols = faceImg.shape[:2]
	nx1 = min(cols-1,max(0,x1)); ny1 = min(rows-1,max(0,y1))
	nx2 = min(cols-1,max(0,x2)); ny2 = min(rows-1,max(0,y2))
	nx1 -= ox1; nx2 -= ox1
	ny1 -= oy1; ny2 -= oy1
	return resized[ny2:ny1+1, nx1:nx2+1]


def applyGlassesFilter(faceImg, glassImg, eyePoints, filterPoints):
	rows,cols = faceImg.shape[:2]
	angle = angleWithHorizontal(eyePoints[0],eyePoints[1])
	factor = getResizingFactorSunglass(filterPoints,eyePoints)
	rect = getRectangle(faceImg, glassImg, eyePoints, filterPoints, factor)
	rect = [(int(x),int(y)) for (x,y) in rect]
	resized = cv2.resize(glassImg,(rect[1][0]-rect[0][0]+1,rect[0][1]-rect[1][1]+1))
	resized = handleResizedFilter(resized,rect,faceImg)
	alpha = resized[:,:,3]/255.0
	resized = resized[:,:,:3]
	newFaceImg = faceImg.copy()
	partFace = newFaceImg[max(rect[1][1],0):min(rect[0][1]+1,rows),max(rect[0][0],0):min(rect[1][0]+1,cols)]
	b1,g1,r1 = cv2.split(partFace)
	b2,g2,r2 = cv2.split(resized)
	bn = (1-alpha)*b1 + alpha*b2
	gn = (1-alpha)*g1 + alpha*g2
	rn = (1-alpha)*r1 + alpha*r2
	img_BGRA = cv2.merge((bn,gn,rn))
	newFaceImg[max(rect[1][1],0):min(rect[0][1]+1,rows),max(rect[0][0],0):min(rect[1][0]+1,cols)] = img_BGRA
	return newFaceImg

def getPoints(img):
	img = img.copy()
	img = img[:,:,:3]
	points = []
	def saveXY(event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONUP:
			points.append((np.float(x),np.float(y)))
	cv2.imshow('image',img)
	cv2.setMouseCallback("image", saveXY)
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			if(len(points)>=2):
				break
	cv2.destroyAllWindows()
	return points

def alg():
	DIR = "../data/filter/"
	filterImg = cv2.imread(DIR + "flower.png", cv2.IMREAD_UNCHANGED)
	print filterImg.shape
	faceImg = cv2.imread(DIR + "../cruz.png", 1)
	filterPoints = getPoints(filterImg)
	facePoints = getPoints(faceImg)
	# facePoints = [(168.0, 301.0), (411.0, 307.0)]
	# filterPoints = [(32.0, 33.0), (574.0, 42.0)]
	newFilterImg = applyGlassesFilter(faceImg,filterImg,facePoints,filterPoints)
	cv2.imshow('New Filter Image',newFilterImg)
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()

alg()