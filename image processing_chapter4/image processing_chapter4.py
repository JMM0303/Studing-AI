import cv2 as cv
import numpy as np
import skimage
import time
import matplotlib.pyplot as plt
import sys

# 에지 검출 (sobel 함수)
img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x=cv.Sobel(gray, cv.CV_32F,1,0,ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F,0,1,ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength=cv.addWeighted(sobel_x, 0.5,sobel_y,0.5,0)

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely',sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()


# 캐니 에지
img=cv.imread('soccer.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny1=cv.Canny(gray, 50,150)
canny2=cv.Canny(gray, 100,200)


cv.imshow('Original',gray)
cv.imshow('Canny1',canny1)
cv.imshow('Canny2', canny2)

cv.waitKey()
cv.destroyAllWindows()


# 경계선 찾기
img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(gray, 100, 200)

contour, hierarchy=cv.findContours (canny, cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

lcontour=[]
for i in range(len(contour)):
    if contour[i].shape[0]>100:
        lcontour.append(contour[i])

cv.drawContours(img, lcontour, -1, (0,255,0),3)

cv.imshow('Original with contours', img)
cv.imshow('Canny', canny)

cv.waitKey()
cv.destroyAllWindows()

# 허프 변환
img=cv.imread('apples.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

apples=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,200,param1=150,param2=20,minRadius=50,maxRadius=120)

for i in apples[0]:
    cv.circle(img, (int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

cv.imshow('Apple detection',img)

cv.waitKey()
cv.destroyAllWindows()

# 슈퍼 화소 분할
img-skimage.data.coffee()
cv.imshow('Coffee image',cv.cvtColor(img,cv.COLOR_RGB2BGR))

slic1=skimage.segmentation.slic(img,compactness=20,n_segments=600)
sp_img1=skimage.segmentation.mark_boundaries(img,slic1)
sp_img1=np.uint8(sp_img1*255.0)

slic2=skimage.segmentation.slic(img.compactness=40,n_segments=600)
sp_img2=skimage.segmentation.mark_boundaries(img,slic2)
sp_img2=np.uint8(sp_img2*255.0)

cv.imshow('Super pixels (compact 20)', cv.cvtColor(sp_img1,cv.COLOR_RGB2BGR))
cv.imshow('Super pixels (compact 40)',cv.cvtColor(sp_img2,cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()


# 최적화 분할
coffee=skimage.data.coffee()

start=time.time()
slic=skimage.segmentation.slic(coffee, compactness=20,n_segments=600,start_label=1)
g=skimage.future.graph.rag_mean_color(coffee, slic,mode='similarity')
ncut=skimage.future.graph.cut_normalized(slic,g)
print(coffee.shape,' Coffee image segmentation ',time.time()-start, 'seconds taken')

marking=skimage.segmentation.mark_boundaries (coffee, ncut)
ncut_coffee=np.uint8(marking*255.0)

cv.imshow('Normalized cut',cv.cvtColor(ncut_coffee,cv.COLOR_RGB2BGR))

cv.waitKey()
cv.destroyAllWindows()


# GrabCut
img-cv.imread('soccer.jpg')
img_show=np.copy(img)

mask=np.zeros((img.shape[0],img.shape[1]), np.uint8)
mask[:,:]=cv.GC_PR_BGD

BrushSiz=9
LColor, RColor=(255,0,0), (0,0,255)

def painting(event,x,y, flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, LColor,-1)
        cv.circle(mask, (x,y), BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, RColor,-1)
        cv.circle(mask, (x,y), BrushSiz,cv.GC_BGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, LColor,-1)
        cv.circle(mask, (x,y), BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, RColor,-1)
        cv.circle(mask, (x,y), BrushSiz,cv.GC_BGD,-1)
        
    cv.imshow('Painting',img_show)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        break

background=np.zeros((1,65), np.float64)
foreground=np.zeros((1,65), np.float64)

cv.grabCut(img, mask, None, background, foreground, 5,cv.GC_INIT_WITH_MASK)
mask2=np.where((mask==cv.GC_BGD) (mask==cv.GC_PR_BGD),0,1).astype('uint8')
grab=img*mask2[:,:,np.newaxis]
cv.imshow('Grab cut image',grab)

cv.waitKey()
cv.destroyAllWindows()


# 영역 특징
orig-skimage.data.horse()
img-255-np.uint8(orig)*255
cv.imshow('Horse', img)

contours, hierarchy=cv.findContours (img,cv.RETR_EXTERNAL, cv-CHAIN_APPROX_NONE)

img2=cv.cvtColor(img,cv.COLOR_GRAY2BGR)
cv.drawContours (img2, contours,-1, (255,0,255),2)
cv.imshow('Horse with contour', img2)

contour=contours[0]

m=cv.moments(contour)
area=cv.contourArea (contour)
cx,cy=m['m10']/m['m00'],m['m01']/m['m00']
perimeter-cv.arcLength(contour, True)
roundness (4.0*np.pi*area)/(perimeter*perimeter)
print('Area=', area, '\nCenter=(',cx,',',cy, ')', '\nPerimeter=', perimeter, '\nRoundness=',roundness)

img3=cv.cvtColor(img,cv.COLOR_GRAY2BGR)

contour_approx=cv.approxPolyDP (contour, 8, True)
cv.drawContours (img3, [contour_approx], -1, (0,255,0),2)

hull=cv.convexHull (contour)
hull-hull.reshape(1, hull.shape[0], hull.shape[2])
cv.drawContours (img3, hull,-1,(0,0,255),2)

cv.imshow('Horse with line segments and convex hull', img3) ④

cv.waitKey()
cv.destroyAllWindows()

