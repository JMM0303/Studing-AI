import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# RGB 컬러 영상을 채널별로 구분해 디스플레이하기
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('original_RGB', img)
cv.imshow('Upper left half', img[0:img.shape[0]//2, 0:img.shape[1]//2])
cv.imshow('Center half', img[img.shape[0]//4:3*img.shape[0]//4,
                           img.shape[1]//4:3*img.shape[1]//4])

cv.imshow('R channel', img[:, :, 2])
cv.imshow('G channel', img[:, :, 1])
cv.imshow('B channel', img[:, :, 0])

cv.waitKey()
cv.destroyAllWindows()


# 이진화 - 실제 영상에서 히스토그램 구하기
img = cv.imread('soccer.jpg')
h = cv.calcHist([img],[2],None,[256],[0,256]) #2번 채널인 R 채널에서 히스토그램 구함
plt.plot(h,color='r',linewidth=1)

# 오츄 알고리즘
img=cv.imread('soccer.jpg')

t, bin_img = cv.threshold(img[:,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
print('오츄 알고리즘이 찾은 최적 임곗값=',t)

cv.imshow('R channel',img[:,:,2])
cv.imshow('R channel binariztion',bin_img)

cv.waitKey()
cv.destroyAllWindows()

# 모폴로지
img=cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(bin_img,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
plt.imshow(b,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se=np.uint8([[0,0,1,0,0],
             [0,1,1,1,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
             [0,0,1,0,0]])

b_dilation-cv.dilate(b,se, 반복 횟수=1)
plt.imshow(b_dilation,cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_erosion=cv.erode(b, se, 반복 횟수=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b_closing=cv.erode(cv.dilate(b, se, 반복 횟수=1), se, 반복 횟수=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 감마 보정
img=cv.imread('soccer.jpg')
img=cv.resize(img, dsize=(0,0), fx=0.25, fy=0.25)

def gamma(f, gamma=1.0):
    f1=f/255.0
    return np.uint8(255*(f1**gamma))

gc-np.hstack((gamma(img,0.5), gamma(img, 0.75), gamma(img, 1.0), gamma(img, 2.0), gamma(img,3.0)))

cv.imshow('gamma gc)

cv.waitKey()
cv.destroyAllWindows()

# 히스토그램 평활화
img=cv.imread('mistyroad.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h=cv.calcHist([gray], [0], None, [256], [0,256])
plt.plot(h,color='r', linewidth=1), plt.show()

equal=cv.equalizeHist(gray)
plt.imshow(equal,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h=cv.calcHist([equal], [0], None, [256], [0,256])
plt.plot(h,color='r',linewidth=1), plt.show()


# 데이터 형과 컨볼루션
img-cv.imread('soccer.jpg')
img-cv.resize(img,dsize=(0,0), fx=0.4,fy=0.4)
gray-cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.putText(gray, 'soccer', (10,20),cv.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
cv.imshow('Original',gray)

smooth-np.hstack((cv.Gaussian Blur (gray, (5,5),0.0),cv.GaussianBlur(gray, (9,9),0.0), cv.GaussianBlur(gray, (15,15),0.0)))
cv.imshow('Smooth', smooth)

femboss-np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

gray16-np.int16(gray)
emboss-np uint8(np.clip(cv.filter2D(gray16,-1, femboss)+128,0,255))
emboss_bad=np.uint8(cv.filter2D(gray16,-1, femboss)+128)
emboss_worse=cv.filter2D(gray,-1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_worse', emboss_worse)

cv.waitKey()
cv.destroyAllWindows()


# 영상 보간
img=cv.imread('rose.png')
patch=img[250:350,170:270,:]

img=cv.rectangle(img, (170,250), (270,350),(255,0,0),3)
patch1=cv.resize(patch,dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

cv.imshow('Original',img)
cv.imshow('Resize nearest', patch1)
cv.imshow('Resize bilinear',patch2)
cv.imshow('Resize bicubic', patch3)

cv.waitKey()
cv.destroyAllWindows()


# OPENCV의 시간효율
def my_cvtGray1(bgr_img):
    g=np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r,c]=0.114*bgr_img[r,c,0]+0.587*bgr_img[r,c,1]+0.299*bgr_img[r,c,2]
    return np.uint8(g)

def my_cvtGray2(bgr_img):
    g=np.zeros([bgr_img.shape[0],bgr_img.shape[1]])
    g=0.114*bgr_img[:,:,0]+0.587*bgr_img[:,:,1]+0.299*bgr_img[:,:,2]
    return np.uint8(g)
    
img=cv.imread('girl_laughing.png')

start=time.time()
my_cvtGray1(img)
print('My time1:',time.time()-start)

start=time.time()
my_cvtGray2(img)
print('My time2:',time.time()-start)

start=time.time()
cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print('OpenCV time:',time.time()-start)
