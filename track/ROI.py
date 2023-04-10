import numpy as np
from matplotlib import pyplot as plt
import cv2

src = cv2.imread("./1.jpg")

ROI = np.zeros(src.shape, np.uint8) #感兴趣区域ROI
proimage0 = src.copy()               #复制原图

"""提取轮廓"""
proimage1=cv2.cvtColor(proimage0,cv2.COLOR_BGR2GRAY)   #转换成灰度图
proimage2=cv2.adaptiveThreshold(proimage1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,7)
contours,hierarchy=cv2.findContours(proimage2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE) #提取所有的轮廓


"""ROI提取"""
cv2.drawContours(ROI, contours, 1,(255,255,255),-1)       #ROI区域填充白色，轮廓ID1
ROI=cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)                  #转换成灰度图
ROI=cv2.adaptiveThreshold(ROI,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,7)                                     #自适应阈值化
imgroi= cv2.bitwise_and(ROI,proimage3)                   #图像交运算 ，获取的是原图处理——提取轮廓后的ROI


titles = ['Original Image', 'proimage1','proimage2', 'proimage3']
images = [proimage0, proimage1, proimage2, proimage3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()


cv2.imshow('roi',roi)
cv2.imshow('imgroi',imgroi)
cv2.waitKey(0)
cv2.destroyAllWindows()