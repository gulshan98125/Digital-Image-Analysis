import numpy as np
import cv2

DIR = "../data/"
filenames = ['arnie.png', 'bush.png']
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

def getDict_points(filename1, filename2, t):
	points1 = getPoints(filename1)
	points2 = getPoints(filename2)
	d = {}
	for p1,p2 in zip(points1, points2):
		x1 = p1[0]
		y1 = p1[1]
		x2 = p2[0]
		y2 = p2[1]
		x = t*x1+(1-t)*x2
		y = t*y1+(1-t)*y2
		p3 = (x,y)
		d[p1] = (p2,p3)
	return d, points1, points2;

def getDictForP3(filename1,filename2,t):
	points1 = getPoints(filename1)
	points2 = getPoints(filename2)
	d ={}
	for p1,p2 in zip(points1, points2):
		x1 = p1[0]
		y1 = p1[1]
		x2 = p2[0]
		y2 = p2[1]
		x = t*x1+(1-t)*x2
		y = t*y1+(1-t)*y2
		p3 = (x,y)
		d[p3] =(p1,p2)
	return d

def topFlatFill(P1,P2,P3,P1lam,P2lam,P3lam, dictA, imgLeft, imgRight,t): #filling using the middle image triangle
	#fill the top flat triangle P1 leftmost point, P2 right most, P3 bottom
	#for a given Y, leftmost X and rightmost X below
	height,width = imgLeft.shape[0]-1,imgLeft.shape[1]-1
	y = P1[1]
	while(y >= P3[1]):
		leftmostX = int(P1[0] + (y-P1[1])*((P3[0]-P1[0])/(P3[1]-P1[1])))
		rightmostX = int(P2[0] + (y-P2[1])*((P3[0]-P2[0])/(P3[1]-P2[1])))
		x = leftmostX
		while(x <= rightmostX):
			Lam1 = ((P2lam[1]-P3lam[1])*(x-P3lam[0])+(P3lam[0]-P2lam[0])*(y-P3lam[1]))/((P2lam[1]-P3lam[1])*(P1lam[0]-P3lam[0])+(P3lam[0]-P2lam[0])*(P1lam[1]-P3lam[1]))
			Lam2 = ((P3lam[1]-P1lam[1])*(x-P3lam[0])+(P1lam[0]-P3lam[0])*(y-P3lam[1]))/((P2lam[1]-P3lam[1])*(P1lam[0]-P3lam[0])+(P3lam[0]-P2lam[0])*(P1lam[1]-P3lam[1]))
			Lam3 = 1-(Lam1+Lam2)

			l1,r1 = dictA[P1lam]	#correspoding triangle point from left image and right image triangles
			l2,r2 = dictA[P2lam]
			l3,r3 = dictA[P3lam]

			#xL,yL from left image corresponding to point x,y
			xL = int(Lam1*(l1[0]) + Lam2*(l2[0]) + Lam3*(l3[0]))
			xL = min(width,xL)
			yL = int(Lam1*(l1[1]) + Lam2*(l2[1]) + Lam3*(l3[1]))
			yL = min(height,yL)

			#xR,yR from right image corresponding to point x,y
			xR = int(Lam1*(r1[0]) + Lam2*(r2[0]) + Lam3*(r3[0]))
			xR = min(width,xR)
			yR = int(Lam1*(r1[1]) + Lam2*(r2[1]) + Lam3*(r3[1]))
			yR = min(height,yR)

			newR = t*(imgLeft[yL,xL][0])+ (1-t)*(imgRight[yR,xR][0])
			newG = t*(imgLeft[yL,xL][1])+ (1-t)*(imgRight[yR,xR][1])
			newB = t*(imgLeft[yL,xL][2])+ (1-t)*(imgRight[yR,xR][2])
			
			if(newR<0):
				newR=0
			if(newG<0):
				newG=0
			if(newB<0):
				newB=0
			finalImg[int(y)][int(x)] = [np.uint8(newR),np.uint8(newG),np.uint8(newB)]
			# print(int(newR),int(newG),int(newB))

			x+=1

		y-=1




def bottomFlatFill(P1,P2,P3,P1lam,P2lam,P3lam,dictA,imgLeft,imgRight,t):
	#fill the bottom flat triangle P1 top point, P2 leftmost, P3 rightmost
	#print("points",P1,P2,P3,P1lam,P2lam,P3lam)
	height,width = imgLeft.shape[0]-1,imgLeft.shape[1]-1
	y = P2[1]
	while(y<=P1[1]):
		leftmostX = int(P1[0] + (y-P1[1])*((P2[0]-P1[0])/(P2[1]-P1[1])))
		#print("leftmostX",leftmostX)
		rightmostX = int(P1[0] + (y-P1[1])*((P3[0]-P1[0])/(P3[1]-P1[1])))
		#print("rightmostX",rightmostX)
		x = leftmostX
		while(x<=rightmostX):
			Lam1 = ((P2lam[1]-P3lam[1])*(x-P3lam[0])+(P3lam[0]-P2lam[0])*(y-P3lam[1]))/((P2lam[1]-P3lam[1])*(P1lam[0]-P3lam[0])+(P3lam[0]-P2lam[0])*(P1lam[1]-P3lam[1]))
			Lam2 = ((P3lam[1]-P1lam[1])*(x-P3lam[0])+(P1lam[0]-P3lam[0])*(y-P3lam[1]))/((P2lam[1]-P3lam[1])*(P1lam[0]-P3lam[0])+(P3lam[0]-P2lam[0])*(P1lam[1]-P3lam[1]))
			Lam3 = 1-(Lam1+Lam2)
			#print("lam",Lam1,Lam2,Lam3)

			l1,r1 = dictA[P1lam]	#correspoding triangle point from left image and right image triangles
			l2,r2 = dictA[P2lam]
			l3,r3 = dictA[P3lam]
			#print("LR",l1,r1,l2,r2,l3,r3)

			#xL,yL from left image corresponding to point x,y
			xL = int(Lam1*(l1[0]) + Lam2*(l2[0]) + Lam3*(l3[0]))
			xL = min(width,xL)
			yL = int(Lam1*(l1[1]) + Lam2*(l2[1]) + Lam3*(l3[1]))
			yL = min(height,yL)

			#xR,yR from right image corresponding to point x,y
			xR = int(Lam1*(r1[0]) + Lam2*(r2[0]) + Lam3*(r3[0]))
			xR = min(width,xR)
			yR = int(Lam1*(r1[1]) + Lam2*(r2[1]) + Lam3*(r3[1]))
			yR = min(height,yR)

			#print("B",xL,yL,xR,yR)

			newR = t*(imgLeft[yL,xL][0])+ (1-t)*(imgRight[yR,xR][0])
			newG = t*(imgLeft[yL,xL][1])+ (1-t)*(imgRight[yR,xR][1])
			newB = t*(imgLeft[yL,xL][2])+ (1-t)*(imgRight[yR,xR][2])
			

			if(newR<0):
				newR=0
			if(newG<0):
				newG=0
			if(newB<0):
				newB=0
			finalImg[int(y)][int(x)] = [np.uint8(newR),np.uint8(newG),np.uint8(newB)]
			# print(int(newR),int(newG),int(newB))

			x+=1
		y+=1



def main(imgLeft,imgRight, filename1, filename2, t):
	d,points1,points2 = getDict_points(filename1,filename2, t)

	triangles = getTriangles(imgLeft,points1)

	#the triangle is of middle image
	for triangle in triangles:
		_,p1 = d[(triangle[0],triangle[1])]
		_,p2 = d[(triangle[2],triangle[3])]
		_,p3 = d[(triangle[4],triangle[5])]

		temp = (p1[1],p2[1],p3[1])
		maxY = max(temp)
		minY = min(temp)

		P1,P2,P3,P4 = None,None,None,None
		pointstuple = (p1,p2,p3)

		indexMax = temp.index(maxY)
		indexMin = temp.index(minY)

		#Triangle convention satisfy karne ke liye
		P1 = pointstuple[indexMax] #Top point
		P2 = pointstuple[indexMin] #bottom point
		P3 = pointstuple[3-(indexMax+indexMin)] #middle point

		#P4 lies in the line joining P1 and P2
		D = getDictForP3(filename1,filename2,t)
		if(P3[1]==P1[1]):
			#flat top triangle
			if(P3[0] < P1[0]):
				topFlatFill(P3,P1,P2,P3,P1,P2,D,imgLeft,imgRight,t)
			else:
				topFlatFill(P1,P3,P2,P1,P3,P2,D,imgLeft,imgRight,t)

		elif(P3[1]==P2[1]):
			#flat bottom triangle
			if(P3[0] < P2[0]):
				bottomFlatFill(P1,P3,P2,P1,P3,P2,D,imgLeft,imgRight,t)
			else:
				bottomFlatFill(P1,P2,P3,P1,P2,P3,D,imgLeft,imgRight,t)
			#fill triangle P1P4P3
		else:
			P4_2 = P3[1]  #Y of P4
			P4_1 = P1[0] + (P4_2 - P1[1])*((P2[0] - P1[0])/(P2[1]-P1[1]))  #X of P4
			P4 = (P4_1,P4_2)

			if(P4[0] > P3[0]):
				bottomFlatFill(P1,P3,P4,P1,P3,P2,D,imgLeft,imgRight,t)
				topFlatFill(P3,P4,P2,P3,P1,P2,D,imgLeft,imgRight,t)
			else:
				bottomFlatFill(P1,P4,P3,P1,P2,P3,D,imgLeft,imgRight,t)
				topFlatFill(P4,P3,P2,P1,P3,P2,D,imgLeft,imgRight,t)

imgLeft = cv2.imread(DIR + filenames[0],1)
imgRight = cv2.imread(DIR + filenames[1],1)

print("working...")
main(imgLeft,imgRight,filenames[0]+"-morph-points.txt",filenames[1] + "-morph-points.txt",0.5)

newImg = np.array(finalImg)
# cv2.imshow('frame',newImg)

# while(True):
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cv2.destroyAllWindows()
cv2.imwrite('../data//output_morph.jpg', newImg)
print("completed")