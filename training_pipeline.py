# prediction : Manually
# gradients computation : Autograd
# loss computation : Pytorch loss
# Parameter updates : Pytorch optimizer

# Design model ( input, output size, forward pass)
# Construct loss and optimizer 
# Training loop 
# - forward pass : compute prediction 
# - backward pass : gradients 
# - update weights 

import torch
import torch.nn as nn

# f = w*x 
# f = 2*x

X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape # 4, 1


input_size = n_features
output_size = n_features
model = nn.Linear( input_size, output_size)

print(f'Prediction before training : f(5) = {model(X_test).item() :.3f}')
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



#Training 


for epoch in range(n_iters):
    #prediction 
    y_pred = model(X)
    
    #loss
    l = loss(Y, y_pred)
    
    #gradients backward pass
    l.backward() #dl/dw
    
    #update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
   
    if epoch%10 == 0:
        [w,b] = model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')