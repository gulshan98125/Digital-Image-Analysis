import reduce as rc
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2

kernel  =     [ 
			[1,4,6,4,1], \
			[4,16,24,16,4], \
			[6,24,36,24,6], \
			[4,16,24,16,4], \
			[1,4,6,4,1]
		  ]
#kernel_multiplier 
K = 1/256.0

levels = 4

def getLuminanceImages(image1,image2):
	rows,cols,_ = image1.shape
	lum_image1 = np.zeros((rows,cols))
	lum_image2 = np.zeros((rows,cols))
	lum_image_toList1 = lum_image1.tolist()
	lum_image_toList2 = lum_image2.tolist()
	for i in xrange(rows):
		for j in xrange(cols):
			bgr = image1[i,j]
			lum_image_toList1[i][j] = (0.299*bgr[2] + 0.587*bgr[1] + 0.114*bgr[0])/255

			bgr2 = image2[i,j]
			lum_image_toList2[i][j] = (0.299*bgr2[2] + 0.587*bgr2[1] + 0.114*bgr2[0])/255
	return np.array(lum_image_toList1),np.array(lum_image_toList2)

def get_feature_array(image):
	big_rows,big_cols,_ = image.shape
	image_lum,_ = getLuminanceImages(image,image)

	image_lum = cv2.copyMakeBorder(image_lum, 2,2,2,2, cv2.BORDER_REPLICATE)

	final_array = []
	for i in xrange(big_rows):
		for j in xrange(big_cols):
			feature_arr = []
			for k in xrange(-2,3):
				for l in xrange(-2,3):
					feature_arr.append(image_lum[i+2+k][j+2+l])
			final_array.append(feature_arr)

	return np.array(final_array)

def getLowestFiltered(A,A_prime,B,levels,color_model):
	min_A =A
	min_A_prime = A_prime
	min_B = B

	for i in xrange(levels):
		min_A = rc.reduce(min_A,kernel,K)
		min_A_prime = rc.reduce(min_A_prime,kernel,K)
		min_B = rc.reduce(min_B,kernel,K)

	output_img = np.copy(min_A)
	output_img_tolist = output_img.tolist()

	feature_array_A = get_feature_array(min_A)
	feature_array_B = get_feature_array(min_B)

	nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(feature_array_A)

	rows,cols,_ = min_A.shape

	if(color_model=="RGB"):
		for i in xrange(rows):
			for j in xrange(cols):
				feature = feature_array_B[i*cols+j]
				best_match_index = nbrs.kneighbors([feature])[1][0][0]
				row_index,col_index = best_match_index/cols, best_match_index%cols
				output_img_tolist[i][j] = min_A_prime[row_index][col_index]

	elif(color_model=="YIQ"):
		for i in xrange(rows):
			for j in xrange(cols):
				feature = feature_array_B[i*cols+j]
				best_match_index = nbrs.kneighbors([feature])[1][0][0]
				row_index,col_index = best_match_index/cols, best_match_index%cols
				bgrA = min_A_prime[row_index][col_index]
				bgrB = min_B[i][j]

				Y = (0.299*bgrA[2] + 0.587*bgrA[1] + 0.114*bgrA[0])/255
				I = (0.596*bgrB[2] - 0.275*bgrB[1] - 0.321*bgrB[0])/255
				Q = (0.212*bgrB[2] - 0.523*bgrB[1] + 0.311*bgrB[0])/255

				r = (Y + ( 0.956 * I) + ( 0.621 * Q)) * 255
				g = (Y - (0.272 * I) - (0.647 * Q)) * 255
				b = (Y - (1.105 * I)  + ( 1.702 * Q)) * 255

				output_img_tolist[i][j] = [b, g, r]

	return np.array(output_img_tolist)



