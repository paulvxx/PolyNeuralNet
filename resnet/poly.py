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

# A Polynomial-Only Residual Network For image classification tasks
class ResPolyNet(nn.Module):
    def __init__(self, input_dim, in_channel, hidden_channel, hidden_dim, coefficients, num_classes, padding=1, pooling_dim=2, learnable=True) -> None:
        super().__init__()
        # kernel size is automatically computed from padding to preserve dimension
        kernel_size = 2 * padding + 1
        self.conv_l1 = nn.Conv2d(in_channel, hidden_channel[0], kernel_size, padding=padding)
        self.poly_activation_l1 = Polynomial(coefficients, learnable=learnable)
        self.conv_l2 = nn.Conv2d(hidden_channel[0], hidden_channel[1], kernel_size, padding=padding)
        self.poly_activation_l2 = Polynomial(coefficients, learnable=learnable)
        self.conv_l3 = nn.Conv2d(hidden_channel[1], hidden_channel[2], kernel_size, padding=padding)
        self.poly_activation_l3 = Polynomial(coefficients, learnable=learnable)
        self.conv_l4 = nn.Conv2d(hidden_channel[2], hidden_channel[3], kernel_size, padding=padding)
        self.poly_activation_l4 = Polynomial(coefficients, learnable=learnable)
        # Should divide the input dimension of the image (length/width)
        self.avg_pooling = torch.nn.AvgPool2d(pooling_dim)
        # Finaly fully connected linear
        incoming_dim = (input_dim[1] * input_dim[2] * hidden_channel[3]) // pooling_dim
        self.linear = torch.nn.Linear(incoming_dim, hidden_dim)
        self.poly_activation_l5 = Polynomial(coefficients, learnable=learnable)
        self.linear2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.conv_l1(x)
        out = self.poly_activation_l1(out)
        out = self.conv_l2(x)
        x_2 = out + x  #residual connection
        out = self.poly_activation_l2(x_2)

        out = self.conv_l3(x)
        out = self.poly_activation_l3(out)        
        out = self.conv_l4(x)
        x_3 = out + x_2  #residual connection
        out = self.poly_activation_l4(x_3)

        out = self.avg_pooling(out)
        out = torch.flatten(out)
        out = self.linear(out)
        return out
