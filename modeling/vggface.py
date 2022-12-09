from numpy.lib.arraysetops import isin
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
import torch

vggf_mean = torch.tensor([0.485, 0.456, 0.406]).float()
vggf_std = torch.tensor([0.229, 0.224, 0.225]).float()

if torch.cuda.is_available():
    vggf_std = vggf_std.cuda()
    vggf_mean = vggf_mean.cuda()

class VggFace(nn.Module):
    def __init__(self):
        super(VggFace, self).__init__()
        self.vggf = self.get_vggf().eval()
        self.mean = vggf_mean.view(-1, 1 ,1)
        self.std = vggf_std.view(-1, 1, 1)

    def forward(self, x):
        return self.vggf(self.normalize_vggf(x))

    @staticmethod
    def get_vggf(last_layer='repeat_1'):
        vggface = InceptionResnetV1(pretrained='vggface2').eval()
        model_list = []

        model_list.append(vggface.conv2d_1a)
        model_list.append(vggface.conv2d_2a)
        model_list.append(vggface.conv2d_2b)
        model_list.append(vggface.maxpool_3a)
        model_list.append(vggface.conv2d_3b)
        model_list.append(vggface.conv2d_4a)
        model_list.append(vggface.conv2d_4b)
        model_list.append(vggface.repeat_1[0])
        model_list.append(vggface.repeat_1[1])
        model_list.append(vggface.repeat_1[2].branch0.conv)

        model = nn.Sequential(*model_list)
        
        return model


    def normalize_vggf(self, image):
        '''
        Expect input in range -1 1
        '''
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from utils.image_processing import normalize_input

    image = Image.open("example/10.jpg")
    image = image.resize((224, 224))
    np_img = np.array(image).astype('float32')
    np_img = normalize_input(np_img)

    img = torch.from_numpy(np_img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    vggf = VggFace()

    feat = vggf(img)

    print(feat.shape)