from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.fpn import FPN101

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, path_to_model, saving_path='../images/'):
        """
        Class for handling pytorch model
        :param path_to_model: str, path to a valid model checkpoint file
        :param saving_path: str, path to a folder where suppressed images will be stored
        """
        self._net = FPN101()
        checkpoint = torch.load(path_to_model)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._net.to(DEVICE)
        self._net.eval()
        self._saving_path = Path(saving_path)
        self._saving_path.mkdir(parents=True, exist_ok=True)
        
    def predict_image(self, img):
        """
        Get model prediction as numpy array using image itself
        :param img: numpy array of shape (H, W, 3) of type int which represents RGB image as input
        :return: numpy array of shape (H, W, 3) of type float
        """
        _input = img.copy().astype(float) / 255.
        _input = cv2.resize(_input, (576, 576))
        _input = torch.Tensor(np.expand_dims(_input.swapaxes(0,2).swapaxes(1,2), axis=0)).to(DEVICE)
        out = self._net(_input)
        out = out[0][0].detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
        
        return out
        
    def predict(self, path_to_img):
        """
        Get model prediction as numpy array using path to image
        :param img: str, path to image to be predicted
        :return: str, path to stored file
        """
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        out = self.predict_image(img)

        where_to_store = str(self._saving_path.joinpath(Path(path_to_img).name)).replace('.png', '_fpn.png').replace('.jpg', '_fpn.jpg').replace('.jpeg', '_fpn.jpeg')
        cv2.imwrite(where_to_store, out * 255.)
        return where_to_store
