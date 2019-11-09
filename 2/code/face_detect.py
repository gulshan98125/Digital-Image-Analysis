from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import scipy

image = cv2.imread('../data/trump.jpg',1)
outArr = np.copy(image)

outList = outArr.tolist()

yCrCbImg = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


rows,cols,_ = image.shape

#color based filtering
for i in xrange(rows):
	for j in xrange(cols):
		r =image[i,j][2]
		g = image[i,j][1]
		b = image[i,j][0]
		y = yCrCbImg[i,j][0]
		cr = yCrCbImg[i,j][1]
		cb = yCrCbImg[i,j][2]

		# if(r>40 and g>40 and b>20 and 
		# 	( (int(max(r,g,b)) - int(min(r,g,b)) )>15 ) and abs(int(r)-int(g))>15 and r>g and r>b ):
		if(
			cb>77 and cb<127 and cr>133 and cr<173 
			and r>40 and g>40 and b>20 and ( (int(max(r,g,b)) - int(min(r,g,b)) )>15 ) and abs(int(r)-int(g))>15 and r>g and r>b
			):
			outList[i][j] = [255,255,255]
		else:
			
			outList[i][j] = [0,0,0]


output = np.uint8(outList)
out_r = cv2.imwrite('../temps/tempO.jpg',output)
output = cv2.imread('../temps/tempO.jpg',1)
# show = cv2.medianBlur(output,3)

eroded = cv2.erode(output, None, iterations=2)
dilated = cv2.dilate(eroded, None, iterations=2)

out_2 = cv2.imwrite('../temps/thresh.jpg',dilated)
thresh = cv2.imread('../temps/thresh.jpg',0)

thresh_out = cv2.imread('../temps/thresh.jpg',1)



# inverted_thresh = cv2.imread('inverted_thresh.jpg',0)
# inverted_thresh_out = cv2.imread('inverted_thresh.jpg',1)
# edges = cv2.Canny(image,100,255)
# edges = cv2.bitwise_not(edges)
# join = cv2.bitwise_and(edges,thresh)
# cv2.imwrite('join.jpg',join)



labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

pix_amount  = []
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 300:
		mask = cv2.add(mask, labelMask)
		# pix_amount.append(numPixels)
# cv2.imwrite('mask.jpg',mask)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
# print(cnts)
bounding_boxes = []
# pix_amount.sort()
# new_pix_amount = pix_amount[-len(cnts):]
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	# ((cX, cY), radius) = cv2.minEnclosingCircle(c)
	area = cv2.contourArea(c)
	ratio_widthToHeight = float(w)/float(h)
	ratio_pixelsToBox = area/float(w*h)
	# print('width','height',w,h)
	# print('pixnum',new_pix_amount[i])
	# print('ratio',ratio_pixelsToBox)

	if(ratio_widthToHeight<.9 and ratio_widthToHeight>0.5 and ratio_pixelsToBox>0.45 and ratio_pixelsToBox<0.9):
		cv2.rectangle(image, (int(x), int(y)),(int(x+w),int(y+h)),(0, 255, 0), 2)
		cv2.rectangle(thresh_out, (int(x), int(y)),(int(x+w),int(y+h)),(0, 255, 0), 2)
		bounding_boxes.append((x,y,w,h))
	# cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

print('number of faces detected = ' + str(len(bounding_boxes)))
cv2.imwrite("../data/OUTPUT_LABELS.jpg", image)
cv2.imwrite("../temps/labels_temp.jpg", thresh_out)
# cv2.imshow('frame1',output)
# cv2.imshow('frame',show)



# # out_e = cv2.imwrite('temp2.jpg',edges)
# # edges = cv2.imread('temp2.jpg',1)


# out3 = cv2.addWeighted(output,1,edges,1,0)
# cv2.imshow('frame3',out3)

# cv2.waitKey(0)

#INVERTED WORK


