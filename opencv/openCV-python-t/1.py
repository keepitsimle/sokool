import numpy as np
import cv2
import matplotlib.pyplot as plt
# img = cv2.imread('0.jpg',0)
# print (img)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.imwrite('light.png',img)

# img = np.ones((0x200,0x200),np.uint8)
# # cv2.line(img,(0,0),(512,512),(255,0,0),5)
# # cv2.imshow('image',img)
# # s = plt.imsave('a.raw',img)
# t = cv2.imread(img)
# print(t)
#
# cv2.waitKey(0)

# light = cv2.imread('0.jpg')
# print('light',light.shape)
# print(light)
# cv2.imshow('image',light)
# print(light.shape)
# cv2.waitKey(0)

# a,b,c  = np.array_split(light,3)
# print(a.shape)

# for i in light:
#     if sum(i)<127*3:
#         i[0]=i[1]=i[2] = 0;
# for i in range(750):
#     for j in range(500):
#         if(sum(light[i][j])<127*3):
#             for k in range(3):
#                 light[i][j][k]=0
# print(light)
#
# cv2.imshow('image',light)
# cv2.waitKey(0)
#
# img_half= cv2.resize(light,(375*2,250*2));
# cv2.imshow('image',img_half)
# cv2.waitKey(0)


'''
    使用函数将BGR->HSV
'''
'''
img_hsv = cv2.cvtColor(light,cv2.COLOR_BGR2HLS)
print(img_hsv.shape)

img_hsv[:,:,0] = (img_hsv[:,:,0]+20)%180

img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HLS2BGR)

cv2.imwrite('turn_green.jpg',img_bgr)

'''
# rgb_value1 =np.sum(light,axis = 2)
# print(rgb_value1.shape)
# # rgb_value = (light[:,:,0]+light[:,:,1]+light[:,:,2])
# np.savetxt('text.txt',rgb_value1,fmt='%3d')
# np.savetxt('r',light[:,:,0],fmt='%3d')
# np.savetxt('g',light[:,:,1],fmt='%3d')
# np.savetxt('b',light[:,:,2],fmt='%3d')
#
# for i in range(750):
#     for j in range(500):
#         if rgb_value1[i,j]<256: #*np.ones((750,500)).all():
#             light[i,j,0] = light[i,j,1]=light[i,j,2] =0
#
#
# cv2.imshow('image',light)
# cv2.waitKey(0)
'''
hist_b = cv2.calcHist([light],[0],None,[256],[0,256])
hist_g = cv2.calcHist([light],[1],None,[256],[0,256])
hist_r = cv2.calcHist([light],[2],None,[256],[0,256])

def gamma_trains(img,gamma):
    gamma_table = [np.power(x/255,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table).astype(np.uint8))

    return cv2.LUT(img,gamma_table)

img_corrected = gamma_trains(light,0.5)
cv2.imwrite('gamma_corr.jpg',img_corrected)

hist_b_corrected = cv2.calcHist([img_corrected],[0],None,[256],[0,256])
hist_g_corrected = cv2.calcHist([img_corrected],[1],None,[256],[0,256])
hist_r_corrected = cv2.calcHist([img_corrected],[2],None,[256],[0,256])



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

pix_hists = [
    [hist_b, hist_g, hist_r],
    [hist_b_corrected, hist_g_corrected, hist_r_corrected]
]

pix_vals = range(256)
for sub_plt, pix_hist in zip([121, 122], pix_hists):
    ax = fig.add_subplot(sub_plt, projection='3d')
    for c, z, channel_hist in zip(['b', 'g', 'r'], [20, 10, 0], pix_hist):
        cs = [c] * 256
        ax.bar(pix_vals, channel_hist, zs=z, zdir='y', color=cs, alpha=0.618, edgecolor='none', lw=0)

    ax.set_xlabel('Pixel Values')
    ax.set_xlim([0, 256])
    ax.set_ylabel('Channels')
    ax.set_zlabel('Counts')

plt.show()
'''

#https://www.zhihu.com/topic/19587715/top-answers
# img = cv2.imread('0.jpg')
# print(img)
#
# M_crop_light = np.array(
#     [
#         [0.5,0,-50],
#         [0,0.5,-100],
#     ],
#     dtype=np.float32)
#
# '''
# x轴剪切变换,y轴剪切15度
# '''
# theta = 15*np.pi/180
# M_sher = np.array(
#     [
#         [1,np.tan(theta),0],
#         [np.tan(theta),1,0]
#     ],dtype=np.float32
# )
#
#
#
# img_light = cv2.warpAffine(img,M_crop_light,(250,375))
# cv2.imwrite('img_light.jpg',img_light)
#
#
# img_light = cv2.warpAffine(img,M_sher,(500,750))
# cv2.imwrite('img_light_sheared.jpg',img_light)



# import numpy as np
# import cv2
# # BGR 三个分量分别是B G R
# img = np.zeros((512,512,3),np.uint8)
#
# cv2.line(img,(0,0),(511,511),(255,0,0),2)
#
# cv2.rectangle(img,(300,0),(450,450),(0,255,0),1)
#
# cv2.circle(img,(400,400),50,(0,0,127),2)
#
# cv2.ellipse(img,(200,200),(20,30),0,0,270,(0,127,0))
#
# pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,0))
#
#
# font = cv2.FONT_HERSHEY_PLAIN
# cv2.putText(img,'sokool',(10,500),font,4,(12,34,45))
#
# cv2.imshow('image',img)
# cv2.waitKey(0)


'''
    将鼠标当画笔
'''

# import cv2
# events = [i for i in dir(cv2) if 'EVENT' in i] # 打印所有的鼠标/键盘事件
# print(events)
'''
['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON',
 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 
 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 
'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']

img = np.zeros((512,512,3),np.uint8)
font = cv2.FONT_HERSHEY_PLAIN
def draw_logo(events,x,y,flags,param):
    if events==cv2.EVENT_LBUTTONDBLCLK:
        cv2.putText(img,'sokool',(100,100),font,2,(100,255,34));

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_logo)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20)&0xFF == 27:
        break
cv2.destroyAllWindows()

'''


'''
 图像的加法:

img_0 = cv2.imread('0.jpg')

cut_M = np.array(
    [
        [1,0,0],
        [0,1,-250]
    ],
    dtype=np.float,
)
cut_img_0 = cv2.warpAffine(img_0,cut_M,(500,500))
# cv2.imshow('image',cut_img_0)
# cv2.waitKey(0)
cv2.imwrite('0.jpg',cut_img_0)
img_1 = cv2.imread('1.jpg')

i_0_1 = cv2.addWeighted(cut_img_0,0.5,img_1,0.5,0)

cv2.imshow('image',i_0_1)
cv2.waitKey(0)
# print(cut_img_0.shape,img_1.shape)

    
'''


'''
    将1.jpg 缩小为原来的1/4,加入遮挡住0.jpg


opencv_1_4_M = np.array(
    [
        [0.5,0,0],
        [0,0.5,0],
    ],dtype=np.float
)

logo = cv2.imread('1.jpg',0)
opencv_1_4 = cv2.warpAffine(logo,opencv_1_4_M,(250,250));
cv2.imwrite('opencv_logo.jpg',opencv_1_4);

'''
'''
img = np.ones((500,500),np.uint8)

# img = cv2.imread(img_arr,0)
# cv2.imshow('1',img)
# print(logo.shape)
print(opencv_1_4.shape)

# cv2.imshow('img',opencv_1_4)

# red = logo[145:340+50+4,0:195+50+4]

img[0:250,0:250] = opencv_1_4
img[250:500,0:250] = opencv_1_4
img[0:250,250:500] = opencv_1_4
img[250:500,250:500] = opencv_1_4

cv2.imshow('1',img)
cv2.waitKey(0)

'''


'''
    利用HSV跟踪提取颜色
        提取蓝色的图案


img = cv2.imread('1.jpg')
print(img)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

mask = cv2.inRange(img_hsv,lower_blue,upper_blue)
mask_not = cv2.bitwise_not(mask) # blue 的取反操作
output = cv2.bitwise_and(img,img,mask=mask_not)

cv2.imshow('mask',mask)
cv2.imshow('output',output)

cv2.waitKey(0)

'''


'''
    图像的透视变换
    就像放大缩小图像一样;
    

img = cv2.imread('1.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[39,185],[39,375],[233,185],[233,375]])

pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(500,500))


cv2.imshow('1',dst)
cv2.waitKey(0)

'''
'''
    简单阈值

img = cv2.imread('1.jpg',0)
ret,threashold1= cv2.threshold(img,127,255,cv2.THRESH_BINARY);
ret,threashold2= cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV);
ret,threashold3= cv2.threshold(img,127,255,cv2.THRESH_TRUNC);
ret,threashold4= cv2.threshold(img,127,255,cv2.THRESH_TOZERO);
ret,threashold5= cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV);


titles = ['original','binary','binary_inv','trunc','tozero','tozero_inv']
images = [img,threashold1,threashold2,threashold3,threashold4,threashold5]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])

plt.show();

'''


'''
    自适应阈值


img = cv2.imread('1.jpg',0)

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

'''


'''
    Otus's 二值化
img = cv2.imread('1.jpg',0)
# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
#（5,5）为高斯核的大小，0 为标准差
blur = cv2.GaussianBlur(img,(5,5),0)
# 阈值一定要设为0！
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,img, 0, th2,blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
'Original Noisy Image','Histogram',"Otsu's Thresholding",
'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# 这里使用了pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用flatten 方法
#ndarray.flat 1-D iterator over an array.
#ndarray.flatten 1-D array copy of the elements of an array in row-major order.
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
'''

'''
    图像平滑
    https://docs.gimp.org/en/plug-in-convmatrix.html
'''
