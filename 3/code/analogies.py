import reduce as rc
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
import pyramid_base as pyramid_base
import analogy_temp as analogies

levels = 4
kappa = 0.5
color_model = "YIQ"
method = "both"

A = cv2.imread('../input/analogies/A.jpg',1)
A_prime = cv2.imread('../input/analogies/A_prime.jpg',1)
B = cv2.imread('../input/analogies/B.jpg',1)


print('getting base B prime')
red_B_prime = pyramid_base.getLowestFiltered(A,A_prime,B,levels,color_model)
print('got reduced B prime')

A_array = [A]
current_image = A
x=1
while(levels-x>=1):
	current_image = rc.reduce(current_image,rc.kernel,rc.k)
	A_array.append(current_image)
	x+=1

B_array = [B]
current_image = B
x=1
while(levels-x>=1):
	current_image = rc.reduce(current_image,rc.kernel,rc.k)
	B_array.append(current_image)
	x+=1

A_prime_array = [A_prime]
current_image = A_prime
x=1
while(levels-x>=1):
	current_image = rc.reduce(current_image,rc.kernel,rc.k)
	A_prime_array.append(current_image)
	x+=1

cv2.imwrite('../input/analogies/pyramid_intermediate/pyramidbase.jpg',red_B_prime)
for i in xrange(levels-1,-1,-1):
	print("starting work on pyramid level %d"%i)
	#YIQ or RGB
	#approximate of both
	red_B_prime = analogies.getAnalogyResult(A_array[i], A_prime_array[i], B_array[i], red_B_prime, i, levels, kappa, color_model, method)
	cv2.imwrite('../input/analogies/pyramid_intermediate/pyramid'+str(i)+'.jpg',red_B_prime)

if(method=="both"):
	cv2.imwrite('../input/analogies/output_both.jpg',red_B_prime)
elif(method=="approximate"):
	cv2.imwrite('../input/analogies/output_approximate.jpg',red_B_prime)