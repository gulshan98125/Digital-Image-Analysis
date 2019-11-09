import numpy as np
import cv2


#provide box number using median cuts
cubeNum =0
globalNum=0

image = ['Pamela.png','random.png','lena.png','test.jpg','PamelaDOT.jpg']
#img_grey = cv2.imread('../data/' + image[0], 0)
img_orig = cv2.imread('../data/' + image[0], 1)

#image = ['Pamela.PNG','random.png','lena.png','test.jpg','Pamela.jpg']
image_orig = cv2.imread('../data/' + image[0],1)

resultList=[]
boxWithPoints = []
ignoreCubes = []

numberOfSplits = 2

for i in range(numberOfSplits):
    resultList.append(0)
    ignoreCubes.append(0)
    boxWithPoints.append([])

avg_r=resultList[:]
avg_g=resultList[:]
avg_b=resultList[:]
boxCount = resultList[:]

row,col,_ = img_orig.shape

#cube = np.zeros(shape=(row,col),dtype=np.int64)

def initiateBox(rows,cols,boxWithPoints):
    for i in xrange(rows):
        for j in xrange(cols):
            boxWithPoints[0].append((i,j))


def splitColoredImage(img):
    r = img[:,:,:1].reshape(img.shape[:-1])
    g = img[:,:,1:2].reshape(img.shape[:-1])
    b = img[:,:,2:].reshape(img.shape[:-1])
    return r,g,b


#for a given box number find range of colors of points lying in that box
def getRange(matR,matG,matB,boxNum,boxWithPoints):
    minValR = 256
    maxValR = 0
    minValG = 256
    maxValG = 0
    minValB = 256
    maxValB = 0
    arr = boxWithPoints[boxNum]
    for i in xrange(len(arr)):
        if(matR[arr[i][0]][arr[i][1]]<minValR):
            minValR = matR[arr[i][0]][arr[i][1]]
        elif(matR[arr[i][0]][arr[i][1]] >= maxValR):
            maxValR = matR[arr[i][0]][arr[i][1]]
        if(matG[arr[i][0]][arr[i][1]]<minValG):
            minValG = matG[arr[i][0]][arr[i][1]]
        elif(matG[arr[i][0]][arr[i][1]] >= maxValG):
            maxValG = matG[arr[i][0]][arr[i][1]]
        if(matB[arr[i][0]][arr[i][1]]<minValB):
            minValB = matB[arr[i][0]][arr[i][1]]
        elif(matB[arr[i][0]][arr[i][1]] >= maxValB):
            maxValB = matB[arr[i][0]][arr[i][1]]

    return minValR,maxValR,minValG,maxValG,minValB,maxValB


#for a given box number find Median of Color given by Mat and points lying in that box
def getMedian(boxNum,mat,boxWithPoints):
    itemslist = []
    arr = boxWithPoints[boxNum]
    for i in xrange(len(arr)):
        if mat[arr[i][0]][arr[i][1]] not in itemslist:
            itemslist.append(mat[arr[i][0]][arr[i][1]])
    itemslist.sort()
    if(len(itemslist)<=1):
        return -1
    else:
        midElement = itemslist[len(itemslist)/2]
        return midElement


#for a given box number, gives the maximum ranged color and its lower and higher value
def whichIsMax(r, g, b, rows, cols, boxNum,boxWithPoints):
    minR,maxR, minG,maxG,minB,maxB = getRange(r, g, b,boxNum,boxWithPoints)
    p = maxR-minR
    q = maxG-minG
    t = maxB-minB
    maxRange = max(p,q,t)
    #red=0,green=1,blue=2
    if(maxRange ==p):
        return 0,minR,maxR
    elif(maxRange == q):
        return 1,minG,maxG
    else:
        return 2,minB,maxB


def splitCube(point,along,cubeNum,globalNum,resultList,resultList2,boxWithPoints):
    arr = boxWithPoints[cubeNum]
    temp_box = []

    if(along==0):
        counter =0
        for i in range(len(arr)):
            poi = arr[i]
            if(point>=r[arr[i][0]][arr[i][1]]):
                boxWithPoints[globalNum+1].append(poi)
                #cube[arr[i][0]][arr[i][1]] = globalNum+1
                resultList[globalNum+1] += 1
                resultList[cubeNum] -= 1
            else:
                temp_box.append(poi)

    elif(along==1):
        counter =0
        for i in range(len(arr)):
            poi = arr[i]
            if(point>=g[arr[i][0]][arr[i][1]]):
                boxWithPoints[globalNum+1].append(poi)
                #cube[arr[i][0]][arr[i][1]] = globalNum+1
                resultList[globalNum+1] += 1
                resultList[cubeNum] -= 1
            else:
                temp_box.append(poi)

    elif(along==2):
        counter =0
        for i in range(len(arr)):
            poi = arr[i]
            if(point>=b[arr[i][0]][arr[i][1]]):
                boxWithPoints[globalNum+1].append(poi)
                #cube[arr[i][0]][arr[i][1]] = globalNum+1
                resultList[globalNum+1] += 1
                resultList[cubeNum] -= 1
            else:
                temp_box.append(poi)
    boxWithPoints[cubeNum] = temp_box


#code run from below

r,g,b = splitColoredImage(img_orig)
rows,cols,_ = img_orig.shape
resultList[0] = rows*cols

initiateBox(rows,cols,boxWithPoints)


print("Working on median cut...")
while(globalNum != numberOfSplits-1):
    resultList2 = resultList[:]
    resultList2.sort()
    resultList2.reverse()
    currentCube = 0
    for i in range(len(resultList2)):
        if(ignoreCubes[resultList.index(resultList2[i])] != 1):
            currentCube = resultList.index(resultList2[i])
            break

    a,l,t = whichIsMax(r,g,b,rows,cols,currentCube,boxWithPoints)
    if(a==0):
        MedianVal = getMedian(currentCube,r,boxWithPoints)
        if(MedianVal==t):
            ignoreCubes[currentCube] = 1
            globalNum +=1
        else:
            if(MedianVal != -1):
                splitCube(MedianVal,0,currentCube,globalNum,resultList,resultList2,boxWithPoints)
                globalNum +=1
            else:
                ignoreCubes[currentCube] = 1
    elif(a==1):
            MedianVal = getMedian(currentCube,g,boxWithPoints)
            if(MedianVal==t):
               ignoreCubes[currentCube] = 1
            else:
                if(MedianVal != -1):
                    splitCube(MedianVal,1,currentCube,globalNum,resultList,resultList2,boxWithPoints)
                    globalNum +=1
                else:
                    ignoreCubes[currentCube] = 1
    elif(a==2):
            MedianVal = getMedian(currentCube,b,boxWithPoints)
            if(MedianVal==t):
                ignoreCubes[currentCube] = 1
            else:
                if(MedianVal != -1):
                    splitCube(MedianVal,2,currentCube,globalNum,resultList,resultList2,boxWithPoints)
                    globalNum +=1
                else:
                    ignoreCubes[currentCube] = 1



for i in xrange(len(boxWithPoints)):
    for j in xrange(len(boxWithPoints[i])):
        avg_r[i] += r[boxWithPoints[i][j][0]][boxWithPoints[i][j][1]]
        avg_g[i] += g[boxWithPoints[i][j][0]][boxWithPoints[i][j][1]]
        avg_b[i] += b[boxWithPoints[i][j][0]][boxWithPoints[i][j][1]]
    boxCount[i] = len(boxWithPoints[i])

for i in xrange(len(boxCount)):
    if(boxCount[i]!=0):
        avg_r[i] /= boxCount[i]
        avg_g[i] /= boxCount[i]
        avg_b[i] /= boxCount[i]

def getNewRGBImage(avg_r,avg_g,avg_b,rows,cols,img):
    newImg = np.copy(img)
    for i in xrange(len(boxWithPoints)):
        for j in xrange(len(boxWithPoints[i])):
            newImg[boxWithPoints[i][j][0]][boxWithPoints[i][j][1]] = [avg_r[i],avg_g[i],avg_b[i]]

    cv2.imwrite('../output/output_median_cut/output.jpg', newImg)
    print("Completed Median cut!")

k = 2
numRegions = 256/(2**k)
#print numRegions
arr = []
for i in xrange(numRegions):
    temp = []
    for j in xrange(numRegions):
        temp.append([-1 for x in xrange(numRegions)])
    arr.append(temp)

def getClosestBox(r_value,g_value,b_value):
    ri,gi,bi = r_value>>k,g_value>>k,b_value>>k
    if(arr[ri][gi][bi]==-1):
        min_distance = pow((r_value-avg_r[0]), 2)+pow((g_value-avg_g[0]), 2)+pow((b_value-avg_b[0]), 2)
        index = 0
        for i in xrange(1,len(avg_r)):
            distance = pow((r_value-avg_r[i]), 2)+pow((g_value-avg_g[i]), 2)+pow((b_value-avg_b[i]), 2)
            if(distance < min_distance):
                min_distance = distance
                index = i
        arr[ri][gi][bi] = index
    return arr[ri][gi][bi]

def getBox(row,col,boxWithPoints):
    for i in xrange(len(boxWithPoints)):
        if (row,col) in boxWithPoints[i]:
            return i

#getNewRGBImage(avg_r,avg_g,avg_b,rows,cols,img_orig)

###################### below starts dithering ###############
#image_out = cv2.imread('../output/output_median_cut/' + image[1],1)
cv2.imwrite('../output/output_dither/original.jpg',image_orig)

def getNewRGBImage2(matR,matG,matB,rows,cols,img):
    newImg = np.copy(img)
    for i in xrange(rows):
        for j in xrange(cols):
            newImg[i][j] = [matR[i][j], matG[i][j], matB[i][j]]
    cv2.imwrite('../output/output_dither/output.jpg', newImg)
    print("Completed Dithering!")


orig_r, orig_g, orig_b = splitColoredImage(image_orig)
#out_r, out_g, out_b = splitColoredImage(image_out)
new_r,new_g,new_b = orig_r.tolist(), orig_g.tolist(), orig_b.tolist()

rows, cols, _ = image_orig.shape
newImg = image_orig.tolist()
medImg = image_orig.tolist()

print("working on Dithering...")
for i in xrange(rows-1):
    for j in xrange(cols-1):
        #closestBoxIndex = cube[i][j]
        closestBoxIndex  = getClosestBox(new_r[i][j],new_g[i][j],new_b[i][j])
        closestBoxIndex2  = getClosestBox(orig_r[i][j],orig_g[i][j],orig_b[i][j])

        errR = int(new_r[i][j]) - int(avg_r[closestBoxIndex])
        errG = int(new_g[i][j]) - int(avg_g[closestBoxIndex])
        errB = int(new_b[i][j]) - int(avg_b[closestBoxIndex])
        
        new_r[i][j]           = avg_r[closestBoxIndex]
        new_r[i][j+1]       = np.uint8(new_r[i][j+1] + int(errR*(3/8.0)))
        new_r[i+1][j]       = np.uint8(new_r[i+1][j] + int(errR*(3/8.0)))
        new_r[i+1][j+1]    = np.uint8(new_r[i+1][j+1] + int(errR/4.0))

        new_g[i][j]           = avg_g[closestBoxIndex]
        new_g[i][j+1]       = np.uint8(new_g[i][j+1] + int(errR*(3/8.0)))
        new_g[i+1][j]       = np.uint8(new_g[i+1][j] + int(errR*(3/8.0)))
        new_g[i+1][j+1]    = np.uint8(new_g[i+1][j+1] + int(errR/4.0))

        new_b[i][j]           = avg_b[closestBoxIndex]
        new_b[i][j+1]       = np.uint8(new_b[i][j+1] + int(errR*(3/8.0)))
        new_b[i+1][j]       = np.uint8(new_b[i+1][j] + int(errR*(3/8.0)))
        new_b[i+1][j+1]    = np.uint8(new_b[i+1][j+1] + int(errR/4.0))

        #print("new r g b", new_r[i][j],new_g[i][j], new_b[i][j])
        newImg[i][j] = [new_r[i][j],new_g[i][j],new_b[i][j]]
        medImg[i][j] = [avg_r[closestBoxIndex2], avg_g[closestBoxIndex2], avg_b[closestBoxIndex2]]

newImg = np.array(newImg)
medImg = np.array(medImg)
cv2.imwrite('../output/output_dither/output.jpg', newImg)
cv2.imwrite('../output/output_median_cut/output.jpg', medImg)

#getNewRGBImage2(orig_r,orig_g,orig_b,rows,cols, image_orig)
