use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::{
    BatchNorm, BatchNormConfig,
    pool::{MaxPool2d, MaxPool2dConfig},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    //conv layers
    conv_1: ConvBlock<B>,
    conv_2: ConvBlock<B>,
    max_pool_1: MaxPool2d,
    conv_3: ConvBlock<B>,
    max_pool_2: MaxPool2d,

    //fc layers
    dropout_1: Dropout,
    fc_1: LinearBlock<B>,
    dropout_2: Dropout,
    fc_2: LinearBlock<B>,
    dropout_3: Dropout,
    output: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv_1 = ConvBlock::new([1, 32], [3, 3], device);
        let conv_2 = ConvBlock::new([32, 64], [3, 3], device);
        let max_pool_1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
        let conv_3 = ConvBlock::new([64, 128], [3, 3], device);
        let max_pool_2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let dropout_1 = DropoutConfig::new(0.5).init();
        let fc_1 = LinearBlock::new(128 * 49, 128, device);
        let dropout_2 = DropoutConfig::new(0.5).init();
        let fc_2 = LinearBlock::new(128, 64, device);
        let dropout_3 = DropoutConfig::new(0.5).init();
        let output = LinearConfig::new(64, 10).with_bias(false).init(device);

        return Self {
            conv_1,
            conv_2,
            max_pool_1,
            conv_3,
            max_pool_2,
            dropout_1,
            fc_1,
            dropout_2,
            fc_2,
            dropout_3,
            output,
        };
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();
        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv_1.forward(x);
        let x = self.conv_2.forward(x);
        let x = self.max_pool_1.forward(x);
        let x = self.conv_3.forward(x);
        let x = self.max_pool_2.forward(x);
        let [batch_size, channels, height, width] = x.dims();

        //linear
        let x = x.reshape([batch_size, channels * height * width]);
        let x = self.dropout_1.forward(x);
        let x = self.fc_1.forward(x);
        let x = self.dropout_2.forward(x);
        let x = self.fc_2.forward(x);
        let x = self.dropout_3.forward(x);

        return self.output.forward(x);
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_stride([1, 1])
            .init(device);
        let norm = BatchNormConfig::new(channels[1]).init(device);
        return Self {
            conv,
            norm,
            activation: Relu::new(),
        };
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        return self.activation.forward(x);
    }
}

#[derive(Module, Debug)]

pub struct LinearBlock<B: Backend> {
    linear: Linear<B>,
    norm: BatchNorm<B, 1>,
    activation: Relu,
}

impl<B: Backend> LinearBlock<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(input, output)
            .with_bias(false)
            .init(device);
        let norm = BatchNormConfig::new(output).init(device);

        return Self {
            linear,
            norm,
            activation: Relu::new(),
        };
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(input);
        let [batch_size, channels] = x.dims();
        let x = x.reshape([batch_size, channels, 1]);
        let x = self.norm.forward(x);
        let x = x.reshape([batch_size, channels]);
        return self.activation.forward(x);
    }
}
