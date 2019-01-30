import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import torch.utils.data

#####################
# Settings
num_examples = 1000000
batch_size = 64
seq_length = 16
input_size = 4
hidden_size = 32
output_size = input_size
num_layers = 2
dropout = 0.0

epochs = 100
lr = 0.0001
#####################


#####################
# Functions
def create_sins(num_points=100):
    amplitude = torch.randn(1)
    frequency = torch.randn(1)/5
    delta = torch.randn(1)
    
    x = [amplitude*torch.sin(frequency*x+delta) for x in range(num_points)]
    y = [amplitude*torch.sin(0.5*frequency*x+delta) for x in range(num_points)]
    
    return x+y

#####################
# Recurrent NN model
class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNNNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.ll = nn.Linear(self.hidden_size, output_size)

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))
        
    def forward(self, X):
        
        h = self.init_hidden(batch_size)
        X, h = self.rnn(X, h) 
        X = self.ll(X)
        
        return X.view(X.size(0),-1)

#####################
# Create dataset
x = torch.tensor([float(x) for x in torch.arange(seq_length*input_size)])
frequency = torch.randn(num_examples)/5
amplitude = torch.add(torch.ones(seq_length*input_size,1), torch.randn(num_examples))
phase = torch.add(torch.ones(seq_length*input_size,1), torch.randn(num_examples))

X = amplitude.t()*torch.sin(torch.mm(frequency.view(-1,1),x.view(1,-1))+phase.t())
Y = amplitude.t()*torch.sin(0.5*torch.mm(frequency.view(-1,1),x.view(1,-1))+phase.t()) # Y is the X with it's frequency divided by 2

# Plot a dataset example
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(X[0,:].squeeze().detach().numpy(), color='r')
ax.plot(Y[0,:].squeeze().detach().numpy(), color='b')

data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y), batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)

#####################
# Train model
model = RNNNetwork(input_size, hidden_size, output_size, num_layers, dropout)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

training_results = {}
for e in range(1,epochs+1):
    
    train_loss = 0.0
    model.train()
    
    optimizer.zero_grad()
  
    for i,(x,y) in enumerate(data_loader,1):
        optimizer.zero_grad()
        y_pred = model(x.view(-1,seq_length,input_size))
        loss = criterion(y_pred,y)
        loss.backward()
        optimizer.step()
    
        train_loss += loss.detach().item()
         
    print('Epoch: {} | Loss: {:.4}'.format(e, train_loss))
    
    if e%10 == 0:
        model.eval()
        training_results[e] = [x, [y,y_pred]]

##########################
# Verify learning process
for epoch in training_results.keys():
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Epoch : {}".format(epoch))
    ax[0].plot(training_results[epoch][0][1,:].squeeze().detach().numpy(), color='b')
    ax[1].plot(training_results[epoch][1][0][1,:].squeeze().detach().numpy(), color='g', ls='dashed')
    ax[1].plot(training_results[epoch][1][1][1,:].squeeze().detach().numpy(), color='r')


#################################
# Test for different input signal

model.eval()
import time

x = torch.tensor([float(x) for x in torch.arange(seq_length*input_size)])
frequency = torch.randn(num_examples)/5
amplitude = torch.add(torch.ones(seq_length*input_size,1), torch.randn(num_examples))
phase = torch.add(torch.ones(seq_length*input_size,1), torch.randn(num_examples))

X = amplitude.t()*torch.sin(torch.mm(frequency.view(-1,1),x.view(1,-1))+phase.t())
Y = amplitude.t()*torch.sin(0.5*torch.mm(frequency.view(-1,1),x.view(1,-1))+phase.t())

data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y), batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0)

start_time = time.time()

X,Y = next(iter(data_loader))
y_pred = model(X.view(-1,seq_length,input_size))

print("Delay : {:.3f} ms".format((time.time()-start_time)*100))

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(X[0,:].squeeze().detach().numpy(), color='b')
ax[1].plot(Y[0,:].squeeze().detach().numpy(), color='g', ls='dashed')
ax[1].plot(y_pred[0,:].squeeze().detach().numpy(), color='r')



