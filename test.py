import torch
import torchvision
from torch import nn , optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

import math
irange = range

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513 , 0.26158784))
])

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid

def loss_function(final_img,residual_img,upscaled_img,com_img,orig_img):

  com_loss = nn.MSELoss(size_average=False)(orig_img, final_img)
  rec_loss = nn.MSELoss(size_average=False)(residual_img,orig_img-upscaled_img)
  
  return com_loss + rec_loss


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    if image.size(0) != 3:
        image = torch.cat((image, image, image), 0)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    
    return image  #assumes that you're using GPU

testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=False, num_workers=2)

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

CHANNELS = 3
HEIGHT = 256
WIDTH = 256
EPOCHS = 20
LOG_INTERVAL = 500

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
   

class End_to_end(nn.Module):
  def __init__(self):
    super(End_to_end, self).__init__()
    
    # Encoder
    self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
    self.bn1 = nn.BatchNorm2d(64, affine=False)
    self.conv3 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
    
    # Decoder
    self.interpolate = Interpolate(size=HEIGHT, mode='bilinear')
    self.deconv1 = nn.Conv2d(CHANNELS, 64, 3, stride=1, padding=1)
    self.deconv2 = []
    for _ in range(18):
        self.deconv2.append(nn.Conv2d(64, 64 ,3))
        self.deconv2.append(nn.BatchNorm2d(64))
        self.deconv2.append(nn.ReLU6())
    self.deconv2 = nn.Sequential(*self.deconv2)
    # self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    # self.bn2 = nn.BatchNorm2d(64, affine=False)
    self.deconv3 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
    
    # self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    # self.bn_n = nn.BatchNorm2d(64, affine=False)

    
    # self.deconv3 = nn.ConvTranspose2d(64, CHANNELS, 3, stride=1, padding=1)
    
    
    self.relu = nn.ReLU()
  
  def encode(self, x):
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.bn1(out)
    return self.conv3(out)
    
  
  def reparameterize(self, mu, logvar):
    pass
  
  def decode(self, z):
    upscaled_image = self.interpolate(z)
    out = self.relu(self.deconv1(upscaled_image))
    out = self.deconv2(out)
    out = self.deconv3(out)
    # for _ in range(10):
    #   out = self.relu(self.deconv_n(out))
    #   out = self.bn_n(out)
    # out = self.deconv3(out)
    final = upscaled_image + out
    return final,out,upscaled_image

    
  def forward(self, x):
    com_img = self.encode(x)
    final,out,upscaled_image = self.decode(com_img)
    return final, out, upscaled_image, com_img, x
    
    
CUDA = torch.cuda.is_available()
if CUDA:
  model = End_to_end().cuda()
else :
  model = End_to_end()
  
  
# model.load_state_dict(torch.load('./net.pth', map_location=torch.device('cpu')))
model1 = torch.load('./net.pth', map_location=torch.device('cpu'))
model = net.load_state_dict(model1['state_dict'])


def test():
  
  model.eval()
  test_loss = 0
  for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        imageName = 'Peppers'
        image = image_loader('./images/'+ imageName + '.jpg')

        # image = make_grid(image)
        final, residual_img, upscaled_image, com_img, orig_im = model(image)
        # if torch.cuda.is_available():
        #     test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).image.cuda()
        # else:
        #     test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).image
            

        n = min(image.size(0), 6)
        print(n);
        comparison = torch.cat([image[:n],
            final[:n].cpu()])
        comparison = comparison.cpu()
#             print(comparison.data)
        save_image(com_img[:n],
                    './processed/compressed/' + imageName + 'Compressed_' + '.jpeg', nrow=n)
        save_image(residual_img[:n],
                    './processed/residual/' + imageName + 'Residual_' + '.jpeg', nrow=n)
        save_image(upscaled_image[:n],
                    './processed/upscaled/' + imageName + 'Upscaled_' + '.jpeg', nrow=n)
        save_image(final[:n],
                    './processed/final/' + imageName + 'Final_' + '.jpeg', nrow=n)
        print("saving the image "+str(n))

  test_loss /= len(test_loader.dataset)
  print('====> Test set loss: {:.4f}'.format(test_loss))
  
test()