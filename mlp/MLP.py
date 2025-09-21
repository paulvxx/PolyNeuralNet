import torch
import torch.nn as nn

# Define a Custom Polynomial Activation Function
# coefficients : A 1d PyTorch Tensor c = [c_0,c_1,c_2,...c_n]  representing polynomial coefficients: c_0*x^n + c_1*x^{n-1} + ... + c_{n-1}*x + c_n
# learnable : Boolean Variable to indicate whether the initialized coefficients should be learned during training, Default: True
class Polynomial(nn.Module):
    def __init__(self, coefficients, learnable=True):
        super().__init__() 
        self.degree = coefficients.shape[0] - 1
        self.coefficients = coefficients
        # Make the coefficients Parameters if True
        if learnable:
            self.coefficients = nn.Parameter(coefficients, requires_grad=True)
        else:
            self.register_buffer("poly_coefficients", coefficients)
    # Use Horner's Method to evaluate the polynomial at a given point
    def forward(self, x):
        res = torch.zeros_like(x)
        for i in range(self.degree + 1):
            res = res * x + self.coefficients[i]
        return res


class CustomPolyMLP(nn.Module):
    def __init__(self, act='relu', scale=1.0, coefficients=None, learnable=True):
        super().__init__() 
        self.activation = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        if act=='poly':
            self.activation = Polynomial(coefficients=coefficients, learnable=learnable)
            self.activation2 = Polynomial(coefficients=coefficients, learnable=learnable)
            torch.nn.init.xavier_uniform_(self.fc1.weight, gain=scale)
            torch.nn.init.xavier_uniform_(self.fc2.weight, gain=scale)
            torch.nn.init.xavier_uniform_(self.fc3.weight, gain=scale)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        return out
