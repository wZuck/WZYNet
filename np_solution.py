# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import numpy as np
import WZYNet
import WZYNet.module as nn
import WZYNet.loss as loss
import WZYNet.optim as optim
# ============== Define Hyperparameters ==============
batch_size = 32
in_dim = 2
out_dim = 1
n_epoch = 100
iterations = 10000

hidden_dim = 10


# ============== Define Dataloader ==============
random_loader = WZYNet.dataloader.proj1_loader(
    batch_size=batch_size, iterations=iterations)

iter_1 = random_loader.__iter__().__next__()
print("X,Y = {}, Z = {}, sin(X)-cos(Y) = {}.".format(iter_1[0][0], iter_1[1][0], np.sin(
    iter_1[0][0][0])-np.cos(iter_1[0][0][1])))


# ============== Define Model ==============
linear_layer_1 = nn.Linear(in_dim=in_dim, out_dim=hidden_dim)
activation_1 = nn.Sigmoid()
linear_layer_2 = nn.Linear(in_dim=hidden_dim, out_dim=out_dim)
model = nn.ModuleSequence(
    [linear_layer_1, activation_1, linear_layer_2])

# ============== Train ==============

criterion = loss.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.1)

loss = 0

for epoch in range(n_epoch):
    for iter, data in enumerate(random_loader):
        optimizer.zero_grad()
        xs, y = data  # 32*2   32*1
        y1 = model(xs)
        loss, gradient = criterion(y1, y)
        model.backward(gradient)
        optimizer.step()
        # if iter % 100 == 0:
        # print('Epoch {}, Iter {}, Loss {}'.format(epoch, iter, loss))
    print('Epoch {},Loss {}'.format(epoch, loss))


# ============== Show PLot ==============

# ------------- The real z=sin(x)-cos(y) surface -------------      FIXME
# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# x, y = np.meshgrid(x, y)
# z = np.sin(x)-np.cos(y)
# WZYNet.show_plot(x, y, z,'Real Surface')

# ------------- WZYNet trained model -------------      FIXME
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
input_as_batch = []
for y_i in y:
    for x_i in x:
        xs = np.array([x_i, y_i])
        input_as_batch.append(xs)
input_as_batch = np.vstack(input_as_batch)
ys = model(input_as_batch)
x, y = np.meshgrid(x, y)
z = ys.reshape([100, 100])
WZYNet.show_plot(x, y, z, 'Hidden neurons = {}'.format(hidden_dim))
