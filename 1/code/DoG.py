import numpy as np
import cv2

image = ['xdog.jpg',"Pamela.png"]
img_orig = cv2.imread('../data/' + image[0], 0)

def dog(k):
	sigma = 1.22
	ksigma = 1.6*sigma
	tau = 1
	epsilon = 0
	blurs = np.array(cv2.GaussianBlur(img_orig,(k,k),sigmaX=sigma,sigmaY=sigma),dtype=float)
	blursk = np.array(cv2.GaussianBlur(img_orig,(k,k),sigmaX=ksigma,sigmaY=ksigma),dtype=float)
	img = blurs-(blursk)
	idx1 = img[:,:] > epsilon
	idx2 = img[:,:] <= epsilon
	img[idx1] = 255
	img[idx2] = 0
	img = np.array(img,dtype=np.uint8)
	return img

def getQuanitzedImage(edges):
	import popularity
	img = cv2.imread('../data/' + image[0], 1)
	quantizedImage = popularity.alg(img)
	idx = edges[:,:] == 255
	quantizedImage[idx] = [255,255,255]
	return quantizedImage

def xdog(k):
	sigma = 1.22
	ksigma = 1.6*sigma
	tau = 1.1
	phi = 1
	epsilon = 0
	blurs = np.array(cv2.GaussianBlur(img_orig,(k,k),sigmaX=sigma,sigmaY=sigma),dtype=float)
	blursk = np.array(cv2.GaussianBlur(img_orig,(k,k),sigmaX=ksigma,sigmaY=ksigma),dtype=float)
	img = blurs-(tau*blursk)
	def f(x):
	    return 0 if x<epsilon else 250+np.tanh(phi*x)
	f = np.vectorize(f)
	img = f(img)
	img = np.array(img,dtype=np.uint8)
	return img

def dog2(k):
	sigma = 1.22
	ksigma = 1.4*sigma
	tau = 1
	epsilon = -1
	rows,cols = img_orig.shape[:2]

	img = np.array(img_orig,dtype=np.float)
	kernel = cv2.getGaussianKernel(k,sigma)
	kernel = kernel.reshape(1,k)
	kernel = np.dot(kernel.T,kernel)
	# blurX = cv2.filter2D(img, -1, kernel.reshape(k))
	blurs = cv2.filter2D(img, -1, kernel)
	
	kernel = cv2.getGaussianKernel(k,ksigma)
	kernel = kernel.reshape(1,k)
	kernel = np.dot(kernel.T,kernel)
	# blurX = cv2.filter2D(img, -1, kernel.reshape(k))
	blursk = cv2.filter2D(img, -1, kernel)

	img = blurs-blursk
	idx1 = img[:,:] > epsilon
	idx2 = img[:,:] <= epsilon
	img[idx1] = 255
	img[idx2] = 0
	img = np.array(img,dtype=np.uint8)
	return img


if __name__=="__main__":
	edges = dog(7)
	quantizedImage = getQuanitzedImage(edges)
	cv2.imshow('frame',quantizedImage)
	while(True):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()