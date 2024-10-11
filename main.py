from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox
from PyQt6.QtWidgets import QMainWindow, QApplication
from qt_material import apply_stylesheet
from CycleGAN_NCNN_Widget import Ui_Form

import numpy as np
import torch
from PIL import Image, ImageQt
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import ncnn


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


class MainApp(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.net = self.load_ncnn_model()
        self.uploaded_image_path = None
        self.converted_image = None

        self.pushButton_upload.clicked.connect(self.upload)
        self.pushButton_convert.clicked.connect(self.convert)
        self.pushButton_download.clicked.connect(self.download)

    def load_ncnn_model(self):
        net = ncnn.Net()
        net.load_param("./assert/hayao.ncnn.param")
        net.load_model("./assert/hayao.ncnn.bin")
        return net

    def upload(self):
        file_dialog = QFileDialog()
        self.uploaded_image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "",
                                                                  "Image Files (*.png *.jpg *.jpeg)")

        if self.uploaded_image_path:
            pixmap = QPixmap(self.uploaded_image_path)
            scene = QGraphicsScene()
            scene.addPixmap(pixmap)
            self.graphicsView_input_img.setScene(scene)
            self.graphicsView_input_img.show()

    def convert(self):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        img = Image.open(self.uploaded_image_path).convert('RGB')
        in0 = preprocess(img).squeeze(0).numpy()

        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0))

            _, out0 = ex.extract("out0")
            out0 = torch.from_numpy(np.array(out0)).unsqueeze(0)
            self.converted_image = Image.fromarray(tensor2im(out0))
            # plt.imshow(self.converted_image)
            # plt.show()

        pixmap = QPixmap.fromImage(ImageQt.ImageQt(self.converted_image))
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        self.graphicsView_output_img.setScene(scene)
        self.graphicsView_output_img.show()


    def download(self):
        if self.converted_image:
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")

            if save_path:
                self.converted_image.save(save_path)
        else:
            QMessageBox.critical(self, "No Image", "No image to save!")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_amber.xml', extra={'font_size': '30px'})

    App = MainApp()
    sys.exit(app.exec())
