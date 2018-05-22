# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# # img = cv2.imread('venv/0.jpg', 1)
# # # print(img)
# # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# # cv2.imshow('image', img)
# # cv2.waitKey(0) #诡异的很,我说怎么没有png生成,原来没有执行完程序,等待键盘输入后结束程序;程序停止在这里了;
# # cv2.destroyAllWindows()#删除自建的窗口
# # cv2.imwrite('lighting.png', img)
#
# # img = cv2.imread('venv/0.jpg', 0)
# # cv2.imshow('image',img)
# # k = cv2.waitKey()
# # if k == 27:
# #     cv2.destroyAllWindows()
# # elif k==ord('s'):
# #     cv2.imwrite('light.png',img)
# #     cv2.destroyAllWindows()
#
# img = cv2.imread('venv/0.jpg', 1)
# plt.imshow(img, cmap='gray', interpolation='nearest')
# # plt.imshow(img)
#
# plt.xticks([])
# plt.yticks([])
# plt.show()

# t  = 1
#
# for i in range(100):
#     t = 1.0/(t+1)
#     print(t)


import numpy as np


X= np.array(
    [
     [3,3,1],
     [4,3,1],
     [1,1,-1],
    ],
    dtype = np.int32,
)

W = np.array(
    [0,0],
    dtype = np.int32
)
b = 0
# print(1*X[0][3]*X[0][0:3])
#
# for i in range(3):
#     while((X[i][0:2].dot(W)+b)*X[i][2]<=0):
#         W +=1*X[i][2]*X[i][0:2]
#         b +=1*X[i][2]
#         print("W,b,i",W,b,i);
#     for j in range(3):
#         while ((X[j][0:2].dot(W) + b) * X[j][2] <= 0):
#                 W += 1 * X[j][2] * X[j][0:2]
#                 b += 1 * X[j][2]
#                 print("W,b,j", W, b, j);


# print("---W---b",W,b)



#
# import matplotlib.pyplot as plt
#
# x1 = np.linspace(-10,10,100)
# x2 = np.linspace(-10,10,100)
#
#
# y = x1+x2+b
#
# plt.plot(x1,x2)
# plt.show()

from sympy.parsing.sympy_parser import parse_expr
from sympy import plot_implicit

ezplot = lambda expr:plot_implicit(parse_expr(expr))


ezplot('2*x1+3*x2+5')










