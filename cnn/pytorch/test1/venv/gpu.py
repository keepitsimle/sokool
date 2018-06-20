
"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
torchvision
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

# torch.manual_seed(1)


'''
转换到GPU上需要将数据和神经网络都转移到cpu上
'''

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# !!!!!!!! Change in here !!!!!!!!! #
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.  # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2), )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2), )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()

# !!!!!!!! Change in here !!!!!!!!! #
cnn.cuda()  # Moves all model parameters and buffers to the GPU.

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        b_x = x.cuda()  # Tensor on GPU
        b_y = y.cuda()  # Tensor on GPU
        # print('step',step);
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            # print("output.size",test_output.size())
            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU max 取列的最大值的下标
            # print("prey.size",pred_y.size())
            print("x,y", torch.sum(pred_y == test_y).cpu().data.numpy() ,test_y.size(0))
            accuracy = torch.sum(pred_y == test_y).cpu().data.numpy()  / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = cnn(test_x[:1000])

# !!!!!!!! Change in here !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:1000], 'real number')

cnt = 0
for i in range(1000):
    if pred_y[i]!=test_y[i]:
        print(i,pred_y[i],test_y[i])
        cnt+=1
print('cnt',cnt)

# x = [[2,3,4,5]]
# x = (torch.tensor(100)/100).data.numpy()
# # y = torch.max(x,1)
# print('y',x)