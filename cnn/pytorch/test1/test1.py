
from __future__ import print_function
import torch

# x= torch.empty(5,3) # Tensors 与 numpy 的数组类似 可以使用gpu进行加速
# print(x)

# x = torch.rand(5,3) #construct a randomly matrix
# print(x)

# x = torch.zeros(5,3,dtype= torch.float)
# print(x)
# print(type(x))

# y = torch.ones(3,3,dtype = torch.long)
#
# print(y)

# q = torch.tensor([5.0,23])  #directly construct a tensor
# print(q)

# w = x.new_ones(5,3,dtype=torch.double) # 使用旧的x构造
# print(w)

# e = torch.randn_like(w,dtype = torch.float)
# print(e)


# print(e.size()) #get its size() return tuple
#
# r = torch.rand(5,3)
# print(type(r))
# add_t = x.add(r)
# print(add_t)

# u = torch.ones(3,3)
# dot_i = u*x
# print(dot_i)

# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py

# o = torch.ones(4,5,requires_grad = True)
# print(o)
#
# p = o+2
# print(p)
# # print(p.grad_fn)
#
# z = p*p*3
# out = z.mean()
# print(z,out)
#
# d = torch.randn(2,2)
# d = ((d*3)/(d-1))
# print(d)
# print(d.requires_grad)
#
# d.requires_grad_(True)
# # #
# print(d.requires_grad)
#
# f = (d*d).sum()
# print(f)

#
# x = torch.ones(2,2,requires_grad = True)
# print(x)
#
# y = x+2
#
# z = y*y*3
#
# out = z.mean()
# print(z,out)
#
# out.backward()
# print(x.grad)
#
# x = torch.randn(3,requires_grad = True)
# y = x*2
# while y.data.norm()<1000: #二范数
#     y = y*2
# print(y)
#
# gradients  = torch.tensor([0.1,1.0,0.0001],dtype= torch.float)
# y.backward(gradients)
# print(x.grad)

import numpy   as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a,1,out=a)
# print(a)
# print(b)

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#
#         t1 = self.conv1(x)
#         print("t1",t1.size(),t1)
#
#         t2 = F.relu(t1)
#         print('t2',t2.size(),t2)
#
#         t3 = F.max_pool2d(t2,(2,2))
#         print('t3',t3.size(),t3)
#
#         t4 = self.conv2(t3)
#         print('t4',t4.size())
#
#         t5 = F.relu(t4)
#         print('t5',t5.size())
#
#         t6 = F.max_pool2d(t5,2)
#         print('t6',t6.size())
#
#         t7 = self.num_flat_features(t6)
#         print('t7',1)
#
#         t8 = t6.view(-1,t7)
#         print('t8',t8.size())
#
#         t9 = self.fc1(t8)
#         print('t9',t9.size())
#
#         t10 = F.relu(t9)
#         print('t10',t10.size())
#
#         t11 = self.fc2(t10)
#         print('t11',t11.size())
#
#         t12 = self.fc3(t11)
#         print('t12',t12.size());
#
#         return t12;
#
#
#
#
#
#         # If the size is a square you can only specify a single number
#         # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         # x = x.view(-1, self.num_flat_features(x))
#         # x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         # return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# print(net)
#
# # params = list(net.parameters())
# # print(len(params))
# # print(params[0].size())  # conv1's .weight
#
# input = torch.ones(1, 1, 5,5)
# out = net(input)
#
# import tensorflow as tf
# import numpy as np
#
# # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
# x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
#
# # 构造一个线性模型
# #
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# # 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# # 初始化变量
# init = tf.initialize_all_variables()
#
# # 启动图 (graph)
# sess = tf.Session()
# sess.run(init)
#
# # 拟合平面
# for step in range(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print( step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

'''
基本使用:
    使用graph来表示计算任务
    必须在session的上下文中执行graph
    使用tensor表示数据
    使用feed和fetch操作变量
'''

# import tensorflow as tf
#
# mat = tf.constant([[3,3]]) #构造了一个op 1*2
# print(mat)
#
# mat2 = tf.constant([[2],[2]])#构造了一个op 2*1
#
# product = tf.matmul(mat,mat2)
#
# # print(product)
#
# session = tf.Session()
# result = session.run(product)
# print(result,mat,mat2)
# session.close()


# state = tf.Variable(0,name="counter")
# one = tf.constant(1)
# import torch
# import numpy as np


# np_data = np.arange(6).reshape((2,3))
# torch_data = torch.from_numpy(np_data) # 将numpy转换成torch
#
# torch2array = torch_data.numpy()
#
# print(np_data)
# print(torch_data)
# print(torch2array)

import torch
from torch.autograd import Variable

tensor = torch.ones(2,3,requires_grad=True)

tensor1 = torch.ones(2,3,requires_grad= True)


y = torch.(tensor,tensor1)



# v_out.backward()
# print(variable.grad)
# print(variable.)


#t_out.backwrad()  # 只有variable 才有backward()
print(tensor.grad)
print(y.grad_fn)
print()

# print(
#     "\ntensor",tensor,
#     "\nvariable",variable,
#     '\ntype of variable',type(variable),
#     "\nt_out",t_out,
#     "\nv_out",v_out,
# )