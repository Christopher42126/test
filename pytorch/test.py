import numpy as np
import torch
import torch.nn as nn


x = np.arange(1, 12, dtype=np.float32).reshape(-1, 1)
y = 2 * x + 3


# 继承nn.module，实现前向传播，线性回归可看作全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, inp):
        out = self.linear(inp)
        return out
    
    
regression_model = LinearRegressionModel(1, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
regression_model.to(device)


epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(regression_model.parameters(), learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    inputs = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).to(device)
    
    optimizer.zero_grad()
    outpus = regression_model(inputs)
    loss = criterion(outpus, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'epoch {epoch}, loss {loss.item()}')
        

predict = regression_model(torch.from_numpy(x).requires_grad_().to(device))

result = predict.to('cpu').detach().numpy()

print(result)

torch.save(regression_model.state_dict(), 'pytorch/model.pkl')
result = regression_model.load_state_dict(torch.load('pytorch/model.pkl'))

