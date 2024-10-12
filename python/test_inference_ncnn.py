import numpy as np
import ncnn
import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    img = Image.open('test.jpg').convert('RGB')
    in0 = preprocess(img).squeeze(0).numpy()


    with ncnn.Net() as net:
        net.load_param("hayao.ncnn.param")
        net.load_model("hayao.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0))

            _, out0 = ex.extract("out0")
            out0 = torch.from_numpy(np.array(out0)).unsqueeze(0)
            output_img = Image.fromarray(tensor2im(out0))
            plt.imshow(output_img)
            plt.show()


if __name__ == "__main__":
    test_inference()
