import numpy as np
import cv2

DIR = "../data/"
filenames = ['trump.jpg', 'cruz.png']
img_orig = cv2.imread(DIR + filenames[0],1)
finalImg = img_orig.tolist()

def getPoints(filename):
	file = open(DIR+filename)
	lines = file.readlines()
	points = []
	for line in lines:
		x,y = [int(s) for s in line.split()]
		points.append((x,y))
	return points

def getTriangles(img, points):
	size = img.shape
	rect = (0,0,size[1],size[0])
	subdiv = cv2.Subdiv2D(rect)
	for point in points:
		subdiv.insert(point)
	triangles = subdiv.getTriangleList()
	newTriangles = []
	for i in xrange(len(triangles)):
		isValid = True
		for j in xrange(len(triangles[i])):
			if(j%2==0):
				p = (triangles[i][j],triangles[i][j+1])
				if(p not in points):
					isValid=False; break
			triangles[i][j] = triangles[i][j]
		if(isValid):
			newTriangles.append(triangles[i])
	return newTriangles

def drawTriangles(img,triangles):
	for triangle in triangles:
		p1 = (triangle[0],triangle[1])
		p2 = (triangle[2],triangle[3])
		p3 = (triangle[4],triangle[5])
		cv2.line(img, p1, p2, (0,255,0),2)
		cv2.line(img, p2, p3, (0,255,0),2)
		cv2.line(img, p3, p1, (0,255,0),2)

def drawPoints(img,points):
	for point in points:
		cv2.circle(img, point, 2, (0,255,0), -1)

if __name__=="__main__":
	
	img = cv2.imread(DIR + filenames[1], 1)

	points = getPoints(filenames[1]+"-points.txt")
	triangles = getTriangles(img,points)
	drawTriangles(img,triangles)
	
	cv2.imshow('frame',img)
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()