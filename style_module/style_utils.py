import torch 
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
from facenet_pytorch import MTCNN

# This detects if cuda is available for GPU training, otherwise it will use CPU
device = torch.device('cpu') #("cuda" if torch.cuda.is_available else "cpu")

# Helper function to scale and transform image into torch tensor
def image_loader(image_name, imsize):
    loader = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
    image = Image.open(image_name)
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device)

# Helper function to scale and transform image into torch tensor
def image_loader_from_BGR(face, imsize):
    # convert to RGB
    face_RGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_RGB)
    #face_img_array = np.array(face_pil)
    loader = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
    # Fake batch dimension required to fit network's input dimensions
    image = loader(face_pil).unsqueeze(0)
    return image.to(device)

# Helper function to show the tensor as a PIL image
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage() 
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

# Custom content loss
# This is looking at the content layer and input's MSE
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Style loss
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Helper class for VGG network
# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# Helper function for the creation of the model
def get_style_model_and_losses(cnn,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    normalization_mean=torch.tensor([0.485, 0.456, 0.406]) #.to(device)
    normalization_std=torch.tensor([0.229, 0.224, 0.225])

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Gradient Descent Algo
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

### MTCNN Helper functions####
def draw(img, boxes):
    for box in boxes:
        cv2.rectangle(img, 
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 0, 255),
                    thickness=2)
    return img

def detect_ROI(boxes):
    """
        Return ROIs as a list with each element being the coordinates for face detection box
        (RowStart, RowEnd, ColStart, ColEnd)
    """
    boxes_copy = boxes
    ROIs = list()
    for box in boxes_copy:
        ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        ROIs.append(ROI)
    return ROIs

def extract_face(ROIs, image):
    faces = []
    for roi in ROIs:
        (startY, endY, startX, endX) = roi
        face = image[startY:endY, startX:endX]
        faces.append(face)
    return faces

def extract_face_shape(faces):
    faces_shapes = []
    for face in faces:
        faces_shapes.append(face.shape)
    return faces_shapes

def expand_face_rect2(boxes, expander):

    # box as (x1,y1,x2,y2)
    boxes_copy = boxes.copy()
    boxes_list = []
    for box in boxes_copy:
        expand_x_by = (box[2] - box[0])*expander
        expand_y_by = (box[3] - box[1])*(expander)
    for box in boxes_copy:
        box[0] = box[0] -  expand_x_by
        box[1] = box[1] -  expand_y_by
        box[2] = box[2] +  expand_x_by
        box[3] = box[3] +  expand_y_by/4
        boxes_list.append(box)
    return np.array(boxes_list)

