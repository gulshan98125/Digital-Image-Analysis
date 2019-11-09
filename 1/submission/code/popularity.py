import numpy as np
import cv2

class region:
	r,g,b,freq = 0.0,0.0,0.0,0
	ind = -1
	def addColor(self,r,g,b):
		self.r = (self.freq*self.r + r)/(self.freq+1)
		self.g = (self.freq*self.g + g)/(self.freq+1)
		self.b = (self.freq*self.b + b)/(self.freq+1)
		self.freq += 1
	def getColor(self):
		return [np.uint8(self.r),np.uint8(self.g),np.uint8(self.b)]


k = 3
numRegions = 256/(2**k)
numColors = 64


def calcFrequencies(img):
	arr = []
	for i in xrange(numRegions):
		temp = []
		for j in xrange(numRegions):
			temp.append([region() for x in xrange(numRegions)])
		arr.append(temp)
	rows,cols,_ = img.shape
	i = 0
	while(i<rows):
		j = 0
		while(j<cols):
			r,g,b = img[i][j]
			ri,gi,bi = r>>k,g>>k,b>>k
			arr[ri][gi][bi].addColor(r,g,b)
			j += 1
		i+=1
	return arr

def getPopularColors(arr2):
	arr = np.array(arr2)
	rc,gc,bc = arr.shape
	arr = list(arr.reshape(rc*gc*bc))
	arr.sort(key=lambda x: x.freq, reverse=True)
	return arr[:numColors]

def getMin(region,popColors):
	minDist,minInd = 99999999.0,-1
	i = 0
	l = len(popColors)
	while(i<l):
		c = popColors[i]
		dist = (region.r-c.r)**2
		dist += (region.g-c.g)**2
		dist += (region.b-c.b)**2
		if(dist<minDist):
			minDist = dist
			minInd = i
		i += 1
	return minInd

def remap(arr,popColors):
	r = 0
	while(r<numRegions):
		g = 0
		while(g<numRegions):
			b = 0
			while(b<numRegions):
				arr[r][g][b].ind = getMin(arr[r][g][b],popColors)
				b+=1
			g+=1
		r+=1

def reCalcImage(img,arr,popColors):
	newImg = img.tolist()
	rows,cols,_ = img.shape
	i = 0
	while(i<rows):
		j = 0
		while(j<cols):
			r,g,b = newImg[i][j]
			ri,gi,bi = r>>k,g>>k,b>>k
			if(arr[ri][gi][bi].ind==-1):
				arr[ri][gi][bi].ind = getMin(arr[ri][gi][bi],popColors)
			newImg[i][j] = popColors[arr[ri][gi][bi].ind].getColor()
			j+=1
		i+=1
	return np.array(newImg)

def alg(img):
	arr = calcFrequencies(img)
	popColors = getPopularColors(arr)
	newImg = reCalcImage(img,arr,popColors)
	return newImg




if __name__=="__main__":
	image = ['Pamela.png','PamelaDOT.jpg']
	print("Working...")
	# img_grey = cv2.imread('../data/' + image[0], 0)
	img_orig = cv2.imread('../data/' + image[0], 1)
	cv2.imshow('frame1',img_orig)

	newImg = alg(img_orig)
	cv2.imshow('frame2',newImg)
	# cv2.imwrite('../output/output_popularity/output.jpg', newImg)
	print("done!")

	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()