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

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        else:
            self.register_buffer("coefficients", coefficients)
    # Use Horner's Method to evaluate the polynomial at a given point
    def forward(self, x):
        res = torch.zeros_like(x)
        for i in range(self.degree + 1):
            res = res * x + self.coefficients[i]
        return res

# Implements a Customized Residual Block
# input_dim : Size of the input dimensions : input = (width * height)
# in_channels : Number of input channels to the Residual Block
# out_channels : Number of output channels to the Residual Block
# coefficients : Polynomial Activation coefficients used in the Residual Block
# learnable : Determines if the coefficients should be learned (True) or not during training,       Default=True
# padding: Number of rows/columns of pixels (per corner / side of the image) to pad with zeroes,    Default=1
# stride: Stride to perform convolution,    Default=1
# Note: The kernel need not be specified since the padding automatically determines how big the kernel should be
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels, coefficients, learnable=True, padding=1, reduce_stride=1):
        super().__init__()
        if input_dim[0] % reduce_stride != 0 or input_dim[1] % reduce_stride != 0:
            raise ValueError(f"The stride reduction value {reduce_stride} must divide the input dimensions {input_dim}.")
        # kernel size is automatically computed from padding to preserve dimension
        kernel_size = 2 * padding + 1
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=reduce_stride)
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        # residual convolution connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=reduce_stride)
        # polynomial activation
        self.poly_activation = Polynomial(coefficients, learnable=learnable)

    def forward(self, x):
        out = self.conv_l1(x)
        out = self.poly_activation(out)
        out = self.conv_l2(out)
        # Residual connection, i.e. x = F(x) + x
        residual = self.residual_conv(x)
        out = out + residual
        out = self.poly_activation(out)
        return out

# Implements the ResNet20 architecture as described by https://www.researchgate.net/figure/ResNet-20-architecture_fig3_351046093
# input_dim : Size of the input dimensions : input = (width * height)
# num_classes : The number of classes used  (classification labels)
# in_channels : The number of input channels used,        Default=3
# coefficients : List of Polynomial Coefficients used,    Default=[1.0, 0.0, 0.0]  (representing x^2)
# learnable : Boolean to indicate if the coefficients should be learned during training,        Default=True 
class ResNet20(nn.Module):
    def __init__(self, input_dim, num_classes, in_channels=3, coefficients=torch.tensor([1.0, 0.0, 0.0]), learnable=True) -> None:
        super().__init__()
        if input_dim[0] % 4 != 0 or input_dim[1] % 4 != 0:
            raise ValueError(f"The image dimensions {input_dim[0]} and {input_dim[1]} must be divisible by 4.")
        
        # kernel size is automatically computed from padding to preserve dimension
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(input_dim, 16, 16, coefficients, learnable)
        self.res2 = ResidualBlock(input_dim, 16, 16, coefficients, learnable)
        self.res3 = ResidualBlock(input_dim, 16, 16, coefficients, learnable)
        self.res4 = ResidualBlock(input_dim, 16, 32, coefficients, learnable, reduce_stride=2)
        self.res5 = ResidualBlock(input_dim, 32, 32, coefficients, learnable)
        self.res6 = ResidualBlock(input_dim, 32, 32, coefficients, learnable)
        self.res7 = ResidualBlock(input_dim, 32, 64, coefficients, learnable, reduce_stride=2)
        self.res8 = ResidualBlock(input_dim, 64, 64, coefficients, learnable)
        self.res9 = ResidualBlock(input_dim, 64, 64, coefficients, learnable)
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=(input_dim[0] // 4, input_dim[1] // 4))
        # Finaly fully connected linear
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)
        out = self.avg_pooling(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out
