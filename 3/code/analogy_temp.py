import reduce as rc
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2

def subtr(a,b):
	if len(a) < len(b):
	    c = b.copy()
	    c[:len(a)] -= a
	else:
	    c = b.copy()
	    c[:] -= a[:len(b)]
	return c

#luminance valued matrix of two equal sized images
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

def get_full_shape_feature_array(image, red_image, red_image_prime):
	big_rows,big_cols,_ = image.shape
	image_lum,_ = getLuminanceImages(image,image)
	red_image_lum, red_image_prime_lum = getLuminanceImages(red_image,red_image_prime)

	image_lum = cv2.copyMakeBorder(image_lum, 2,2,2,2, cv2.BORDER_REPLICATE)
	# not considering image_prime
	red_image_lum = cv2.copyMakeBorder(red_image_lum, 1,1,1,1, cv2.BORDER_REPLICATE)
	red_image_prime_lum = cv2.copyMakeBorder(red_image_prime_lum, 1,1,1,1, cv2.BORDER_REPLICATE)

	final_array = []
	for i in xrange(big_rows):
		for j in xrange(big_cols):
			feature_arr = []
			for k in xrange(-1,2):
				for l in xrange(-1,2):
					feature_arr.append(red_image_lum[i/2+1+k][j/2+1+l])
					feature_arr.append(red_image_prime_lum[i/2+1+k][j/2+1+l])
			for k in xrange(-2,3):
				for l in xrange(-2,3):
					feature_arr.append(image_lum[i+2+k][j+2+l])
			final_array.append(feature_arr)

	return np.array(final_array)

#return F according to the synthesized image
def getF(image_lum, red_image_lum, image_prime_lum, red_image_prime_lum, i, j, synth_tracker,rows,cols):
	image_lum = cv2.copyMakeBorder(image_lum, 2,2,2,2, cv2.BORDER_REPLICATE)
	red_image_lum = cv2.copyMakeBorder(red_image_lum, 1,1,1,1, cv2.BORDER_REPLICATE)
	red_image_prime_lum = cv2.copyMakeBorder(red_image_prime_lum, 1,1,1,1, cv2.BORDER_REPLICATE)

	feature_arr = []
	for k in xrange(-1,2):
		for l in xrange(-1,2):
			feature_arr.append(red_image_lum[i/2+k][j/2+l])
			feature_arr.append(red_image_prime_lum[i/2+k][j/2+l])
	for k in xrange(-2,3):
		for l in xrange(-2,3):
			feature_arr.append(image_lum[i+k][j+l])
	# for k in xrange(-2,0):
	# 	for l in xrange(-2,3):
	# 		if((i+k>=0) and (j+l>=0) and (j+l<=cols-1) and synth_tracker[i+k][j+l]):
	# 			feature_arr.append(image_prime_lum[i+k][j+l])
	# if((i-2>=0) and synth_tracker[i-2][j]):
	# 	feature_arr.append(image_prime_lum[i-2][j])
	# if((i-1>=0) and synth_tracker[i-1][j]):
	# 	feature_arr.append(image_prime_lum[i-1][j])
	return feature_arr

def bestCoherenceMatch(row_index,col_index,synth_tracker, B_lum, red_B_lum, output_image_lum, red_B_prime_lum, A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, rows, cols, S):
	r_star = (row_index-2,col_index-2)
	min_difference = np.inf
	Fl_q = getF(B_lum, red_B_lum, output_image_lum, red_B_prime_lum, row_index, col_index, synth_tracker,rows,cols)

	for k in xrange(-2,0):
		for l in xrange(-2,3):
			if((row_index+k>=0) and (col_index+l>=0) and (col_index+l<=cols-1) and synth_tracker[row_index+k][col_index+l]):
				#applying only when neighbour exists and is synthesized
				argument = np.array(S[(row_index+k,col_index+l)]) + np.array((row_index,col_index)) - np.array((row_index+k, col_index+l))
				if(argument[0] >=0 and argument[1] >=0 and argument[0]<=rows-1 and argument[1]<=cols-1):
					Fl_argument = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, argument[0], argument[1], synth_tracker,rows,cols)
					norm = np.linalg.norm(subtr(np.array(Fl_argument),np.array(Fl_q)))**2
					if(norm<min_difference):
						min_difference=norm
						r_star = (row_index+k,col_index+l)

	for k in xrange(0,1):
		for l in xrange(-2,0):
			if((row_index+k>=0) and (col_index+l>=0) and (col_index+l<=cols-1) and synth_tracker[row_index+k][col_index+l]):
				#applying only when neighbour exists and is synthesized
				argument = np.array(S[(row_index+k,col_index+l)]) + np.array((row_index,col_index)) - np.array((row_index+k, col_index+l))
				if(argument[0] >=0 and argument[1] >=0 and argument[0]<=rows-1 and argument[1]<=cols-1):
					Fl_argument = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, argument[0], argument[1], synth_tracker,rows,cols)
					norm = np.linalg.norm(subtr(np.array(Fl_argument),np.array(Fl_q)))**2
					if(norm<min_difference):
						min_difference=norm
						r_star = (row_index+k,col_index+l)

	try:
		result = np.array(S[r_star]) + np.array((row_index,col_index)) - np.array(r_star)
	except:
		result = (-1,-1)
	return tuple(result),Fl_q

def getAnalogyResult(A,A_prime,B,red_B_prime,pyramid_level,levels,kappa, method, case):
	#method is either YIQ or RGB, case is whether approximate or both
	rows,cols,_ = A.shape
	output_image_np = np.copy(B)
	output_image = output_image_np.tolist()

	synth_tracker = np.zeros((rows,cols))
	synth_tracker = synth_tracker.tolist()
	output_image_lum, A_prime_lum = getLuminanceImages(np.array(output_image), A_prime)

	S = {} #dictionary for storing indexes

	temp_dict = {}

	full_feature_arr_A = get_full_shape_feature_array(A, rc.reduce(A,rc.kernel,rc.k), rc.reduce(A_prime,rc.kernel,rc.k))
	full_feature_arr_B = get_full_shape_feature_array(B, rc.reduce(B,rc.kernel,rc.k), red_B_prime)

	nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(full_feature_arr_A)
	B_lum, B_prime_lum = getLuminanceImages(B,np.array(output_image))
	red_B_lum, red_B_prime_lum = getLuminanceImages(rc.reduce(B,rc.kernel,rc.k), red_B_prime)

	A_lum, A_prime_lum = getLuminanceImages(A,A_prime)
	red_A_lum, red_A_prime_lum = getLuminanceImages(rc.reduce(A,rc.kernel,rc.k), rc.reduce(A_prime,rc.kernel,rc.k))

	if(case=="approximate"):
		if(method=="YIQ"):
			for i in xrange(rows):
				for j in xrange(cols):
					full_feature = full_feature_arr_B[(i)*(cols)+j]
					try:
					  index = temp_dict[tuple(full_feature)]
					except:
						index = nbrs.kneighbors([full_feature])[1][0][0]
						temp_dict[tuple(full_feature)] = index
					
					Papp = (index/cols, index%cols)
					S[(i,j)] = Papp
					synth_tracker[i][j] = 1
					
					Y = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					bgr = B[i][j]
					I = (0.596*bgr[2] - 0.275*bgr[1] - 0.321*bgr[0])/255
					Q = (0.212*bgr[2] - 0.523*bgr[1] + 0.311*bgr[0])/255

					r = (Y + ( 0.956 * I) + ( 0.621 * Q)) * 255
					g = (Y - (0.272 * I) - (0.647 * Q)) * 255
					b = (Y - (1.105 * I)  + ( 1.702 * Q)) * 255

					# output_image[i][j] = A_prime[S[(i,j)][0]][S[(i,j)][1]]
					output_image[i][j] = [b,g,r]
					output_image_lum[i][j] = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					print("pyramid_level %d, of size (%d,%d)"%(pyramid_level,rows,cols),i,j)

		elif(method=="RGB"):
			for i in xrange(rows):
				for j in xrange(cols):
					full_feature = full_feature_arr_B[(i)*(cols)+j]
					try:
					  index = temp_dict[tuple(full_feature)]
					except:
						index = nbrs.kneighbors([full_feature])[1][0][0]
						temp_dict[tuple(full_feature)] = index
					
					Papp = (index/cols, index%cols)
					S[(i,j)] = Papp
					synth_tracker[i][j] = 1
					output_image[i][j] = A_prime[S[(i,j)][0]][S[(i,j)][1]]
					output_image_lum[i][j] = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					print("pyramid_level %d, of size (%d,%d)"%(pyramid_level,rows,cols),i,j)


	elif(case=="both"):
		if(method=="YIQ"):
			for i in xrange(rows):
				for j in xrange(cols):
					full_feature = full_feature_arr_B[(i)*(cols)+j]
					try:
					  index = temp_dict[tuple(full_feature)]
					except:
						index = nbrs.kneighbors([full_feature])[1][0][0]
						temp_dict[tuple(full_feature)] = index
					
					Papp = (index/cols, index%cols)
					Pcoh,Fl_q = bestCoherenceMatch(i,j,synth_tracker, B_lum, red_B_lum, output_image_lum, red_B_prime_lum, A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, rows, cols, S)

					if(Pcoh[0] < 0 or Pcoh[1] < 0 or Pcoh[0]>=rows or Pcoh[1]>=cols):
						S[(i,j)] = Papp
					else:
						Fl_Papp = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, Papp[0], Papp[1], synth_tracker,rows,cols)
						d_app = np.linalg.norm(subtr(np.array(Fl_Papp),np.array(Fl_q)))**2

						Fl_Pcoh = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, Pcoh[0], Pcoh[1], synth_tracker,rows,cols)

						d_coh = np.linalg.norm(subtr(np.array(Fl_Pcoh), np.array(Fl_q)))**2

						if(d_coh <= d_app*(1+ kappa*(2**(-pyramid_level)))):
							S[(i,j)] = Pcoh
						else:
							S[(i,j)] = Papp
					# S[(i,j)] = Papp
					synth_tracker[i][j] = 1
					
					Y = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					bgr = B[i][j]
					I = (0.596*bgr[2] - 0.275*bgr[1] - 0.321*bgr[0])/255
					Q = (0.212*bgr[2] - 0.523*bgr[1] + 0.311*bgr[0])/255

					r = (Y + ( 0.956 * I) + ( 0.621 * Q)) * 255
					g = (Y - (0.272 * I) - (0.647 * Q)) * 255
					b = (Y - (1.105 * I)  + ( 1.702 * Q)) * 255

					# output_image[i][j] = A_prime[S[(i,j)][0]][S[(i,j)][1]]
					output_image[i][j] = [b,g,r]
					output_image_lum[i][j] = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					print("pyramid_level %d, of size (%d,%d)"%(pyramid_level,rows,cols),i,j)

		elif(method=="RGB"):
			for i in xrange(rows):
				for j in xrange(cols):
					full_feature = full_feature_arr_B[(i)*(cols)+j]
					try:
					  index = temp_dict[tuple(full_feature)]
					except:
						index = nbrs.kneighbors([full_feature])[1][0][0]
						temp_dict[tuple(full_feature)] = index
					
					Papp = (index/cols, index%cols)
					Pcoh,Fl_q = bestCoherenceMatch(i,j,synth_tracker, B_lum, red_B_lum, output_image_lum, red_B_prime_lum, A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, rows, cols, S)

					if(Pcoh[0] < 0 or Pcoh[1] < 0 or Pcoh[0]>=rows or Pcoh[1]>=cols):
						S[(i,j)] = Papp
					else:
						Fl_Papp = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, Papp[0], Papp[1], synth_tracker,rows,cols)
						d_app = np.linalg.norm(subtr(np.array(Fl_Papp),np.array(Fl_q)))**2

						Fl_Pcoh = getF(A_lum, red_A_lum, A_prime_lum, red_A_prime_lum, Pcoh[0], Pcoh[1], synth_tracker,rows,cols)

						d_coh = np.linalg.norm(subtr(np.array(Fl_Pcoh), np.array(Fl_q)))**2

						if(d_coh <= d_app*(1+ kappa*(2**(-pyramid_level)))):
							S[(i,j)] = Pcoh
						else:
							S[(i,j)] = Papp
					# S[(i,j)] = Papp
					synth_tracker[i][j] = 1
					output_image[i][j] = A_prime[S[(i,j)][0]][S[(i,j)][1]]
					output_image_lum[i][j] = A_prime_lum[S[(i,j)][0]][S[(i,j)][1]]
					print("pyramid_level %d, of size (%d,%d)"%(pyramid_level,rows,cols),i,j)

	

	return np.array(output_image)