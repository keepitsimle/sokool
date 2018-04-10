import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('venv/0.jpg', 1)
# # print(img)
# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image', img)
# cv2.waitKey(0) #诡异的很,我说怎么没有png生成,原来没有执行完程序,等待键盘输入后结束程序;程序停止在这里了;
# cv2.destroyAllWindows()#删除自建的窗口
# cv2.imwrite('lighting.png', img)

# img = cv2.imread('venv/0.jpg', 0)
# cv2.imshow('image',img)
# k = cv2.waitKey()
# if k == 27:
#     cv2.destroyAllWindows()
# elif k==ord('s'):
#     cv2.imwrite('light.png',img)
#     cv2.destroyAllWindows()

img = cv2.imread('venv/0.jpg', 1)
plt.imshow(img, cmap='gray', interpolation='nearest')
# plt.imshow(img)

plt.xticks([])
plt.yticks([])
plt.show()