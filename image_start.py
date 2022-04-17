#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

import torch

if not torch.cuda.is_available():
    print('GPU not available.')

import fastai
from deoldify.visualize import *

torch.backends.cudnn.benchmark = True

colorizer = get_image_colorizer(artistic=False)

source_url = 'https://sites-cf.mhcache.com/e/1/WC1BbXotQ29udGVudC1TaGEyNTY9VU5TSUdORUQtUEFZTE9BRCZYLUFtei1DcmVkZW50aWFsPUFLSUFYRDZIUllIRUlUVzRHV0VCJTJGMjAyMTExMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjExMTI5VDA1MDAwMFomWC1BbXotRXhwaXJlcz02MDQ4MDAmWC1BbXotU2lnbmF0dXJlPTE5MzNmM2Y2ZTg3N2EyM2RmOGE0YmMyYTljYzg1Yzg0MzE0Mjg4NDViZDFmMmIyMjBjODU4OWFkMjQ3ZjE1NGU%3D/125/997/0282/500001_789024c81m1i7p1a64c93a_A.jpg' #@param {type:"string"}
render_factor = 35  #@param {type: "slider", min: 7, max: 40}
watermarked = True #@param {type:"boolean"}

if source_url is not None and source_url !='':
    image_path = colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
    # show_image_in_notebook(image_path)
else:
    print('Provide an image url and try again.')

# for i in range(10,40,2):
#     colorizer.plot_transformed_image('test_images/image.png', render_factor=i, display_render_factor=True, figsize=(8,8))