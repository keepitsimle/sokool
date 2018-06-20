import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 500
LR = 0.001
DOWNLOADED_MNIST = False

if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOADED_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download= DOWNLOADED_MNIST,
)

print(train_data.train_data.size());
print(train_data.train_labels.size());

# plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
# print(train_data.train_labels[0])
# plt.show()

if __name__ == '__main__':
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers= 4,
    )

    test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
    test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255  # 归一化除以255
    test_y = test_data.test_labels[:2000]
    print('test_size',test_x.size())
    print('test_y.size()',test_y.size())


    class cNN(nn.Module):
        def __init__(self):
            super(cNN,self).__init__()
            self.conv1 = nn.Sequential( # input (1,28,28)
                nn.Conv2d(
                    in_channels = 1,
                    out_channels= 16,
                    kernel_size= 5,
                    stride= 1,
                    padding=2,
                ),                    # (16,28,28)  14 = (height+2*padding-kernel_size+1)/stride
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  #(16,14,14)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(  #(16 14,14)
                    in_channels=16,
                    out_channels= 32,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ), # (32,(14+2*2-5+1)/2,7) (32,14,14)
                nn.ReLU(),
                nn.MaxPool2d(2), #(32,7,7)
            )
            self.out = nn.Linear(in_features=32*7*7,out_features=10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0),-1)
            output = self.out(x)
            return output,x

    cnn = cNN()
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()




    from matplotlib import cm
    try: from sklearn.manifold import TSNE;HAS_SK = True
    except:HAS_SK = False;


    def plot_with_labels(lowDWeights,labels):
        plt.cla()
        X,Y = lowDWeights[:,0],lowDWeights[:,1]
        for x,y,s in zip(X,Y,labels):
            c = cm.rainbow(int(255*s)/9);
            plt.text(x,y,s,backgroundcolor=c,fontsize=9);
            plt.xlim(X.min(), X.max());
            plt.ylim(Y.min(), Y.max());
            plt.title('Visualize last layer');
            plt.show();
            plt.pause(0.01)


    plt.ion()

    for epoch in range(EPOCH):
        for step,(b_x,b_y) in enumerate(train_loader):
            output = cnn(b_x)[0]
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if step%50==0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = float(sum(pred_y == test_y)) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
    plt.ioff()

    test_output,_ =cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')










