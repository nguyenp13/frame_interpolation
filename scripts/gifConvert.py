__author__ = 'Robert'
from images2gif import writeGif
from PIL import Image
import os

file_names = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
#['animationframa.png', 'animationframb.png', ...] "

images = [Image.open(fn) for fn in file_names]

print writeGif.__doc__

filename = "my_gif.GIF"
writeGif(filename, images, duration=0.2)