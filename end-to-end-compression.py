from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
import os

import torch
import torchvision
from torch import nn , optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

import math
irange = range

accelerator = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")


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
    
    
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513 , 0.26158784))
])

trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=img_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=img_transform)
# testset = 

test_loader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=False, num_workers=2)

# PARAMETERS

CHANNELS = 3
HEIGHT = 32
WIDTH = 32
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
  
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(final_img,residual_img,upscaled_img,com_img,orig_img):

  com_loss = nn.MSELoss(size_average=False)(orig_img, final_img)
  rec_loss = nn.MSELoss(size_average=False)(residual_img,orig_img-upscaled_img)
  
  return com_loss + rec_loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        if torch.cuda.is_available():
            final, residual_img, upscaled_image, com_img, orig_im = model(data.cuda())
        else:
            final, residual_img, upscaled_image, com_img, orig_im = model(data)
        loss = loss_function(final, residual_img, upscaled_image, com_img, orig_im)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
          


def test(epoch):
  
  model.eval()
  test_loss = 0
  for i, (data, _) in enumerate(test_loader):
        print(i, epoch)
        data = Variable(data, volatile=True)
        final, residual_img, upscaled_image, com_img, orig_im = model(data)
        if torch.cuda.is_available():
            test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).data.cuda()
        else:
            test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).data
            
        if epoch == EPOCHS:
            # save_image(final.data[0],'reconstruction_final',nrow=8)
            # save_image(com_img.data[0],'com_img',nrow=8)
            n = min(data.size(0), 6)
            print("saving the image "+str(n))
            comparison = torch.cat([data[:n],
                final[:n].cpu()])
            comparison = comparison.cpu()
    #             print(comparison.data)
            save_image(com_img[:n].data,
                        './test/compressed_' + str(epoch) +'.png', nrow=n)
            save_image(comparison.data,
                        './test/reconstruction_' + str(epoch) +'.png', nrow=n)

  test_loss /= len(test_loader.dataset)
  print('====> Test set loss: {:.4f}'.format(test_loss))
  
  
for epoch in range(1, EPOCHS+1):
    # train(epoch)
    # test(epoch)
    if epoch == EPOCHS:
      pass
      

# torch.save(model.state_dict(), './net.pth')

model.load_state_dict(torch.load('./net.pth', map_location=torch.device('cpu')))

def save_images():
  epoch = EPOCHS
  model.eval()
  test_loss = 0
  for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        # if torch.cuda.is_available():
        #     final, residual_img, upscaled_image, com_img, orig_im = model(data.cuda())
        # else:
        final, residual_img, upscaled_image, com_img, orig_im = model(data)
        test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).data
        # if i == 3:
#             save_image(final.data[0],'reconstruction_final',nrow=8)
#             save_image(com_img.data[0],'com_img',nrow=8)
        # n = min(data.size(0), 6)
        print("saving the image "+str(i))
        comparison = torch.cat([data[:i],
            final[:i].cpu()])
        comparison = comparison.cpu()
#             print(comparison.data)
        save_image(com_img[:1].data,
                        './test/compressed_' + str(i) +'.png', nrow=i)
        save_image(final[:1].data,
                    './test/final_' + str(epoch) +'.png', nrow=i)
        save_image(orig_im[:1].data,
                    'original_' + str(epoch) +'.png', nrow=i)


  test_loss /= len(test_loader.dataset)
  print('====> Test set loss: {:.4f}'.format(test_loss))

save_images()