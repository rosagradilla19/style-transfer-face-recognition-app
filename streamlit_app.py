import streamlit as st
import time
from style_module import *
from facenet_pytorch import MTCNN
import torch 
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image
import time

mtcnn = MTCNN()

st.title('Neural Style Transfer')

st.markdown("This application will perform neural style tranfer on the face from the image you provide")

########################## Ask user for content/style images #############################################
uploaded_content = st.file_uploader(label='Please upload an image to use as the content image', type='jpg')
if uploaded_content is not None:
  # convert the file to an opencv image
  file_bytes = np.asarray(bytearray(uploaded_content.read()), dtype=np.uint8)
  uploaded_content_cv = cv2.imdecode(file_bytes, 1)
  # display the image
  st.image(uploaded_content_cv,   use_column_width=True, channels='BGR')

uploaded_style = st.file_uploader(label='Please upload an image to use as the style image' )
if uploaded_style is not None:
  # convert the file to an opencv image
  file_bytes = np.asarray(bytearray(uploaded_style.read()), dtype=np.uint8)
  uploaded_style_cv = cv2.imdecode(file_bytes, 1)
  # display the image
  st.image(uploaded_style_cv,  use_column_width=True, channels='BGR')


##################################
# This detects if cuda is available for GPU training, otherwise it will use CPU

  device = torch.device('cpu')

# Variable Initialization
# MTCNN initializtion
  mtcnn = MTCNN()

########### Experimental Variables
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
imsize = 224
num_steps = 200

############# Loading images
# This is the image to extract face from
cv_image = None
if uploaded_content is not None:
    cv_image = uploaded_content_cv

# Style image for cv2.imshow
if uploaded_style is not None:
    style_image_display = uploaded_style_cv

# Detect face with MTCNN, organize ROI box, extract face array and face shapes
    boxes_orig, probs = mtcnn.detect(cv_image, landmarks=False)

# Expand ROI, extract faces and face shapes
    boxes = expand_face_rect2(boxes_orig, .25)
    ROIs = detect_ROI(boxes) #(startY, endY, startX, endX)
    faces = extract_face(ROIs, cv_image)
    faces_shapes = extract_face_shape(faces)

# Load BGR face image into torch tensor
    content_img = image_loader_from_BGR(faces[0], imsize)
    style_image_pil = Image.open(uploaded_style)

def image_loader_from_app(image, imsize):
    loader = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
    # Fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device)

if uploaded_style is not None:
  style_img = image_loader_from_app(style_image_pil, imsize)

#assert style_img.size() == content_img.size(), \
#    "we need to import style and content images of the same size"

################ RUN STYLE TRANSFER ####################
if uploaded_style is not None:
# Importing the model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Starting image we are modifying and output
    input_img = content_img.clone()

# This will run the neural style transfer
def run_style_transfer(model,  style_losses, content_losses, input_img, num_steps=num_steps,
                       style_weight=1000000, content_weight=1):
    ### Run the style transfer.
    optimizer = get_input_optimizer(input_img)

    run = [0]
    i = 0
    while run[0] <= num_steps:
        
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            # This is many layers (in the paper)
            for sl in style_losses:
                style_score += sl.loss

            # This is actually one layer (in the paper)
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            # Print progress every 50 runs
            if run[0] % 50 == 0:
                st.text(str("run {}:".format(run)))
                st.text(str('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item())))

            return style_score + content_score

        optimizer.step(closure)

    

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

# Output is a torch tensor, we need to convert to numpy -> cv2 image
def output_to_cv2(output):
    image = output.clone().detach()
    image = image.squeeze(0)
    image = T.ToPILImage(mode='RGB')(image)
    # from BGR to RGB
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) 
    return image

# Replace face

def replace_face(ROIs, styled_face_masks,orig_face, faces_shapes, original_image=cv_image):
    cv_image_copy = np.copy(cv_image)
    for roi, face, face_shape, orig_face in zip(ROIs, styled_face_masks, faces_shapes, orig_face):
        (startY, endY, startX, endX) = roi
        styled_face = cv2.resize(face, dsize=(face_shape[1],face_shape[0]))
        orig_face = cv2.resize(orig_face, dsize=(face_shape[1], face_shape[0]))
        # Overlay mask with original
        dst = np.where(styled_face==0, orig_face, styled_face)
        cv_image_copy[startY:endY, startX:endX] = dst
    return cv_image_copy

# Start computation button and computation
computation_started = False
if st.button('Start computation'):
     computation_started = True
     st.write('Starting style transfer computation....')
    
     model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

     # Finally, run the algorithm
     output = run_style_transfer(model, style_losses, content_losses, input_img, num_steps) #outpus is [B x C x H x W]


########## OUTPUT #########

     image_normalized = output_to_cv2(output= output)

############### FOREGROUND EXTRACTION ###########
if computation_started:
    dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def segment(net, img,  show_orig=False, dev='cpu'):
    """ 
    This function preprocesses the input image, passes it throught
    the network, converts output into a 2D image with is then 
    converted to a binary mask where 0=not-person 1=person
    """
    trf = T.Compose([
                    T.ToTensor(), 
                    T.Normalize(mean = [0.485, 0.456, 0.406], 
                                std = [0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    binary_mask = np.where(om == 15, 1,0)
    # mask where 0 is not-person, and 1 is person
    return binary_mask

def compute_output(mask, ROIs):    
    # reshape mask and make it 3-channels
    mask2_reshaped = cv2.resize(mask2, (faces_shapes[0][1], faces_shapes[0][0]))

    mask_3c = np.zeros_like(orig_face)
    mask_3c[:,:,0] = mask2_reshaped
    mask_3c[:,:,1] = mask2_reshaped
    mask_3c[:,:,2] = mask2_reshaped

    # reshape styled_face
    styled_face_reshaped = cv2.resize(styled_face, (faces_shapes[0][1], faces_shapes[0][0]))

    masked_img = np.where(mask_3c == 0, orig_face, styled_face_reshaped)

    cv_image_copy = np.copy(cv_image)
    for roi in ROIs:
        (startY, endY, startX, endX) = roi
        cv_image_copy[startY:endY, startX:endX] = masked_img
    return cv_image_copy

if computation_started:
    orig_face = faces[0]
    styled_face = image_normalized.copy()
    size = styled_face.shape[:2]

    binary_mask = segment(dlab, orig_face)
    mask2 = binary_mask.astype(float)
    mask2 = cv2.resize(mask2, size)

    output_image = compute_output(mask=mask2, ROIs=ROIs)

# Display output
    st.image(output_image,  use_column_width=True, channels='BGR')


     
