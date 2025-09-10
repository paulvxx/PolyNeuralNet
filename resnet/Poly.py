# Define a Custom Polynomial Activation Function
# coefficients : A 1d PyTorch Tensor c = [c_0,c_1,c_2,...c_n]  representing polynomial coefficients: c_0*x^n + c_1*x^{n-1} + ... + c_{n-1}*x + c_n
# learnable : Boolean Variable to indicate whether the initialized coefficients should be learned during training, Default: True
class Polynomial(nn.Module):
    def __init__(self, coefficients, learnable=True) -> None:
        super().__init__() 
        self.degree = coefficients.shape[0] - 1
        self.coefficients = coefficients
        # Make the coefficients Parameters if True
        if learnable:
            self.coefficients = nn.Parameter(coefficients, requires_grad=True)
    # Use Horner's Method to evaluate the polynomial at a given point
    def forward(self, x):
        res = torch.zeros_like(x)
        for i in range(self.degree + 1):
            res = res * x + self.coefficients[i]
        return res
