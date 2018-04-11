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
img = cv2.imread('0.jpg')

M_crop_light = np.array(
    [
        [0.5,0,-50],
        [0,0.5,-100],
    ],
    dtype=np.float32)

img_light = cv2.warpAffine(img,M_crop_light,(250,375))
cv2.imwrite('img_light.jpg',img_light)
