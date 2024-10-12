import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.padconv2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(7,7), out_channels=64, padding=(3,3), padding_mode='reflect', stride=(1,1))
        self.model_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=256, track_running_stats=False)
        self.model_3 = nn.ReLU()
        self.model_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.model_5 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=128, track_running_stats=False)
        self.model_6 = nn.ReLU()
        self.model_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.model_8 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_9 = nn.ReLU()
        self.padconv2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_10_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_10_conv_block_3 = nn.ReLU()
        self.padconv2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_10_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_11_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_11_conv_block_3 = nn.ReLU()
        self.padconv2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_11_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_5 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_12_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_12_conv_block_3 = nn.ReLU()
        self.padconv2d_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_12_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_13_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_13_conv_block_3 = nn.ReLU()
        self.padconv2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_13_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_14_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_14_conv_block_3 = nn.ReLU()
        self.padconv2d_10 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_14_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_11 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_15_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_15_conv_block_3 = nn.ReLU()
        self.padconv2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_15_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_16_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_16_conv_block_3 = nn.ReLU()
        self.padconv2d_14 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_16_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_15 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_17_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_17_conv_block_3 = nn.ReLU()
        self.padconv2d_16 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_17_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.padconv2d_17 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_18_conv_block_2 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_18_conv_block_3 = nn.ReLU()
        self.padconv2d_18 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='reflect', stride=(1,1))
        self.model_18_conv_block_6 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=64, track_running_stats=False)
        self.model_19 = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=128, output_padding=(1,1), padding=(1,1), stride=(2,2))
        self.model_20 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=128, track_running_stats=False)
        self.model_21 = nn.ReLU()
        self.model_22 = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=64, output_padding=(1,1), padding=(1,1), stride=(2,2))
        self.model_23 = nn.InstanceNorm2d(affine=False, eps=0.000010, num_features=256, track_running_stats=False)
        self.model_24 = nn.ReLU()
        self.padconv2d_19 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(7,7), out_channels=3, padding=(3,3), padding_mode='reflect', stride=(1,1))
        self.model_27 = nn.Tanh()

        archive = zipfile.ZipFile('hayao.pnnx.bin', 'r')
        self.padconv2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_0.bias', (64), 'float32')
        self.padconv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_0.weight', (64,3,7,7), 'float32')
        self.model_4.bias = self.load_pnnx_bin_as_parameter(archive, 'model.4.bias', (128), 'float32')
        self.model_4.weight = self.load_pnnx_bin_as_parameter(archive, 'model.4.weight', (128,64,3,3), 'float32')
        self.model_7.bias = self.load_pnnx_bin_as_parameter(archive, 'model.7.bias', (256), 'float32')
        self.model_7.weight = self.load_pnnx_bin_as_parameter(archive, 'model.7.weight', (256,128,3,3), 'float32')
        self.padconv2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_1.bias', (256), 'float32')
        self.padconv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_1.weight', (256,256,3,3), 'float32')
        self.padconv2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_2.bias', (256), 'float32')
        self.padconv2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_2.weight', (256,256,3,3), 'float32')
        self.padconv2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_3.bias', (256), 'float32')
        self.padconv2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_3.weight', (256,256,3,3), 'float32')
        self.padconv2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_4.bias', (256), 'float32')
        self.padconv2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_4.weight', (256,256,3,3), 'float32')
        self.padconv2d_5.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_5.bias', (256), 'float32')
        self.padconv2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_5.weight', (256,256,3,3), 'float32')
        self.padconv2d_6.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_6.bias', (256), 'float32')
        self.padconv2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_6.weight', (256,256,3,3), 'float32')
        self.padconv2d_7.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_7.bias', (256), 'float32')
        self.padconv2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_7.weight', (256,256,3,3), 'float32')
        self.padconv2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_8.bias', (256), 'float32')
        self.padconv2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_8.weight', (256,256,3,3), 'float32')
        self.padconv2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_9.bias', (256), 'float32')
        self.padconv2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_9.weight', (256,256,3,3), 'float32')
        self.padconv2d_10.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_10.bias', (256), 'float32')
        self.padconv2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_10.weight', (256,256,3,3), 'float32')
        self.padconv2d_11.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_11.bias', (256), 'float32')
        self.padconv2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_11.weight', (256,256,3,3), 'float32')
        self.padconv2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_12.bias', (256), 'float32')
        self.padconv2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_12.weight', (256,256,3,3), 'float32')
        self.padconv2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_13.bias', (256), 'float32')
        self.padconv2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_13.weight', (256,256,3,3), 'float32')
        self.padconv2d_14.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_14.bias', (256), 'float32')
        self.padconv2d_14.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_14.weight', (256,256,3,3), 'float32')
        self.padconv2d_15.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_15.bias', (256), 'float32')
        self.padconv2d_15.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_15.weight', (256,256,3,3), 'float32')
        self.padconv2d_16.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_16.bias', (256), 'float32')
        self.padconv2d_16.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_16.weight', (256,256,3,3), 'float32')
        self.padconv2d_17.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_17.bias', (256), 'float32')
        self.padconv2d_17.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_17.weight', (256,256,3,3), 'float32')
        self.padconv2d_18.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_18.bias', (256), 'float32')
        self.padconv2d_18.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_18.weight', (256,256,3,3), 'float32')
        self.model_19.bias = self.load_pnnx_bin_as_parameter(archive, 'model.19.bias', (128), 'float32')
        self.model_19.weight = self.load_pnnx_bin_as_parameter(archive, 'model.19.weight', (256,128,3,3), 'float32')
        self.model_22.bias = self.load_pnnx_bin_as_parameter(archive, 'model.22.bias', (64), 'float32')
        self.model_22.weight = self.load_pnnx_bin_as_parameter(archive, 'model.22.weight', (128,64,3,3), 'float32')
        self.padconv2d_19.bias = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_19.bias', (3), 'float32')
        self.padconv2d_19.weight = self.load_pnnx_bin_as_parameter(archive, 'padconv2d_19.weight', (3,64,7,7), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.padconv2d_0(v_0)
        v_2 = self.model_2(v_1)
        v_3 = self.model_3(v_2)
        v_4 = self.model_4(v_3)
        v_5 = self.model_5(v_4)
        v_6 = self.model_6(v_5)
        v_7 = self.model_7(v_6)
        v_8 = self.model_8(v_7)
        v_9 = self.model_9(v_8)
        v_10 = self.padconv2d_1(v_9)
        v_11 = self.model_10_conv_block_2(v_10)
        v_12 = self.model_10_conv_block_3(v_11)
        v_13 = self.padconv2d_2(v_12)
        v_14 = self.model_10_conv_block_6(v_13)
        v_15 = (v_9 + v_14)
        v_16 = self.padconv2d_3(v_15)
        v_17 = self.model_11_conv_block_2(v_16)
        v_18 = self.model_11_conv_block_3(v_17)
        v_19 = self.padconv2d_4(v_18)
        v_20 = self.model_11_conv_block_6(v_19)
        v_21 = (v_15 + v_20)
        v_22 = self.padconv2d_5(v_21)
        v_23 = self.model_12_conv_block_2(v_22)
        v_24 = self.model_12_conv_block_3(v_23)
        v_25 = self.padconv2d_6(v_24)
        v_26 = self.model_12_conv_block_6(v_25)
        v_27 = (v_21 + v_26)
        v_28 = self.padconv2d_7(v_27)
        v_29 = self.model_13_conv_block_2(v_28)
        v_30 = self.model_13_conv_block_3(v_29)
        v_31 = self.padconv2d_8(v_30)
        v_32 = self.model_13_conv_block_6(v_31)
        v_33 = (v_27 + v_32)
        v_34 = self.padconv2d_9(v_33)
        v_35 = self.model_14_conv_block_2(v_34)
        v_36 = self.model_14_conv_block_3(v_35)
        v_37 = self.padconv2d_10(v_36)
        v_38 = self.model_14_conv_block_6(v_37)
        v_39 = (v_33 + v_38)
        v_40 = self.padconv2d_11(v_39)
        v_41 = self.model_15_conv_block_2(v_40)
        v_42 = self.model_15_conv_block_3(v_41)
        v_43 = self.padconv2d_12(v_42)
        v_44 = self.model_15_conv_block_6(v_43)
        v_45 = (v_39 + v_44)
        v_46 = self.padconv2d_13(v_45)
        v_47 = self.model_16_conv_block_2(v_46)
        v_48 = self.model_16_conv_block_3(v_47)
        v_49 = self.padconv2d_14(v_48)
        v_50 = self.model_16_conv_block_6(v_49)
        v_51 = (v_45 + v_50)
        v_52 = self.padconv2d_15(v_51)
        v_53 = self.model_17_conv_block_2(v_52)
        v_54 = self.model_17_conv_block_3(v_53)
        v_55 = self.padconv2d_16(v_54)
        v_56 = self.model_17_conv_block_6(v_55)
        v_57 = (v_51 + v_56)
        v_58 = self.padconv2d_17(v_57)
        v_59 = self.model_18_conv_block_2(v_58)
        v_60 = self.model_18_conv_block_3(v_59)
        v_61 = self.padconv2d_18(v_60)
        v_62 = self.model_18_conv_block_6(v_61)
        v_63 = (v_57 + v_62)
        v_64 = self.model_19(v_63)
        v_65 = self.model_20(v_64)
        v_66 = self.model_21(v_65)
        v_67 = self.model_22(v_66)
        v_68 = self.model_23(v_67)
        v_69 = self.model_24(v_68)
        v_70 = self.padconv2d_19(v_69)
        v_71 = self.model_27(v_70)
        return v_71

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def test_inference():
    net = Model()
    net.eval()

    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt

    img_path = 'test.jpg'
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)

    res = net(img)
    # print(res)
    
    output_img = tensor2im(res)
    plt.imshow(output_img)
    plt.show()

    return res

if __name__ == "__main__":
    test_inference()