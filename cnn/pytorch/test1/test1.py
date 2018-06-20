
# from __future__ import print_function
# import torch

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

# import numpy   as np
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

# import torch
# from torch.autograd import Variable

# tensor = torch.ones(2,3,requires_grad=True)
#
# tensor1 = torch.ones(2,3,requires_grad= True)
#
#
# y = torch.add(tensor,tensor1)



# v_out.backward()
# print(variable.grad)
# print(variable.)


#t_out.backwrad()  # 只有variable 才有backward()
# print(tensor.grad)
# print(y.grad_fn)
# print()

# print(
#     "\ntensor",tensor,
#     "\nvariable",variable,
#     '\ntype of variable',type(variable),
#     "\nt_out",t_out,
#     "\nv_out",v_out,
# )

# a = torch.randn(2,2)
# a = ((a*3)/(a-1))
# print(a.requires_grad)
# a.requires_grad_(True)
#
# print(a.requires_grad)
#
# b = (a*a).sum()
# print(b.grad_fn)

'''
x = torch.ones(2,2,requires_grad= True)

y = x+2

z = y*y*3
out = z.mean()
print('x.grad_fn',x.grad_fn)  # x 没有x.grad_fn
print('y.grad_fn',y.grad_fn)  # x 有 requires_grad  接着 x的一套都有grad_fn
print('z.grad_fn',z.grad_fn)
print('out.grad_fn',out.grad_fn)

print(z,out)

out.backward() #
print(x.grad)
print(y.gard) #为什么y没有grad
'''

# import matplotlib.pyplot as plt
# import torch.nn.functional as F


# x = torch.linspace(-5,5,200)
#
# x_np = x.data.numpy()
#
# print(x,x_np)
#
# y_rule = F.relu(x).data.numpy()
#
# y_sigmoid = F.sigmoid(x).data.numpy()
#
# y_tanh = F.tanh(x).data.numpy()
#
# y_softplus = F.softplus(x).data.numpy()

'''
plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_rule,c='red',label='relu')
plt.ylim(-1,50)
plt.legend(loc='relu')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim(-1,1)
plt.legend(loc='sigmoid')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='blue',label='tanh')
plt.ylim(-1,2)
plt.legend(loc='tanh')


plt.subplot(224)
plt.plot(x_np,y_softplus,c='green',label='softplus')
plt.ylim(-1,5)
plt.legend(loc='softplus')

plt.show()
'''
'''
t = torch.unsqueeze(torch.linspace(-1,1,10),dim=1) # t shape = (100,1)
y = t.pow(4) + .01 *torch.rand(t.size())
print(t,'\n',y,'\n',t.pow(2))

# plt.scatter(t.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,in_feature,hidden,out_feature):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(in_feature,hidden)
        self.out = torch.nn.Linear(hidden,out_feature)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(1,10,1)
print(net)

plt.ion()
# plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_func = torch.nn.MSELoss()

for i in range(1000):
    predictor = net(t)

    loss = loss_func(predictor,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%5==0:
        plt.cla()
        plt.scatter(t.data.numpy(),y.data.numpy())
        plt.plot(t.data.numpy(),predictor.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'loss=%4f'%loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.3)
plt.ioff()
plt.show()
'''

'''
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.3*torch.rand(x.size())

x,y = Variable(x,requires_grad=False), Variable(y,requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    optimizer = torch.optim.SGD(net1.parameters(),lr=0.3)
    lossFunc = torch.nn.MSELoss()

    for  t in range(100):
        pre = net1(x)
        loss = lossFunc(pre,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),pre.data.numpy(),'r-',lw=5)

    torch.save(net1,'net.pkl')
    torch.save(net1.state_dict(),'net_params.pkl')

def restore_net():
    net2 = torch.load('net.pkl')
    pre = net2(x)

    plt.subplot(132)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)

def restore_params():
    net3 = net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    pre = net3(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
    plt.show()

save()

restore_net()

restore_params()

'''



# import torch
# import torch.utils.data as Data
#
#
# BATCH_SIZE = 5
#
# x = torch.linspace(1,10,10)
# y = torch.linspace(10,1,10)
#
# torch_dataset = Data.TensorDataset(x,
#                                   y)
#
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size= BATCH_SIZE,
#     shuffle=False,
#     num_workers=2,
# )
#
# for epoch in range(3):
#     for step,(batch_x,batch_y) in enumerate(loader):
#         print('epoch',epoch,'step',step,'batch_x',batch_x.numpy(),'batch_y',batch_y.numpy())

#
# import torch
# import torch.utils.data as Data
#
# torch.manual_seed(1)    # reproducible
#
# BATCH_SIZE = 5
#
# x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
# y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
#
# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(
#     dataset=torch_dataset,      # torch TensorDataset format
#     batch_size=BATCH_SIZE,      # mini batch size
#     shuffle=True,               # random shuffle for training
#     num_workers=1,              # subprocesses for loading data
# )
#
# for epoch in range(3):   # train entire dataset 3 times
#     for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
#         # train your data...
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())


import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 10
EPOCH = 20

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y  = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))


# plt.plot(x.numpy(),y.numpy())
# plt.show()


torch_data = Data.TensorDataset(x,y)

loader = Data.DataLoader(
    dataset = torch_data,
    batch_size=BATCH_SIZE,
    shuffle= True,
    num_workers= 2,
)


class nNet(torch.nn.Module):
    def __init__(self):
        super(nNet,self).__init__()
        self.hidden = torch.nn.Linear(1,200) #这是一个对象来自 完全构造的一个对象
        self.pre = torch.nn.Linear(200,1)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.pre(x)
        return x

if __name__ == '__main__':
    net_sgd = nNet()
    net_momemtum = nNet()
    net_adagrad = nNet()
    net_adam = nNet()


    net_ed = [net_sgd,net_momemtum,net_adagrad,net_adam]

    optimizer_sgd = torch.optim.SGD(net_sgd.parameters(),lr=LR)
    optimizer_Momentum = torch.optim.SGD(net_momemtum.parameters(),lr=LR,momentum=0.8)
    optimizer_adagrad = torch.optim.Adagrad(net_adagrad.parameters(),lr=LR)
    optimizer_adam = torch.optim.Adam(net_adam.parameters(),lr=LR,betas=(0.9,0.99))


    optimizer_ed = [optimizer_sgd,optimizer_Momentum,optimizer_adagrad,optimizer_adam]

    losses_history  = [[],[],[],[]]

    loss_func = torch.nn.MSELoss()

    for i in range(EPOCH):
        print('epoch:',i)
        for step,(bx,by) in enumerate(loader): #每一个epoch将一组数据全部遍历完成
            for net,optimizer,l_his in zip(net_ed,optimizer_ed,losses_history):
                out  = net(bx)
                loss = loss_func(out,by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l_his.append(loss.data.numpy())

    print(losses_history[0].__len__())
    labels = ['sgd','momentun','adgrad','adam']
    print('loss_history',losses_history)
    for i,l_his in enumerate(losses_history):
        plt.plot(l_his,label = labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0,0.2)
    plt.show()


