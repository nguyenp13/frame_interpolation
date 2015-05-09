import numpy as np
import Image
import skimage
import skimage.io
import pdb

from util import *

def main():
    imgNum = 5
    spacing = 20

    path = './images/blocks/out/patchMatch/'
    img = np.array(Image.open(path+'frame_0.png')).astype('float')
    height = img.shape[0]
    width = img.shape[1]
    print img.shape
    output = np.ones((height, width*imgNum+spacing*(imgNum-1), img.shape[2]))

    for i in range(imgNum):
        inputname = path + 'frame_8' + str(i) + '.png'
        print inputname
        tmp = np.array(Image.open(inputname)).astype('float')
        output[0:height,i*width+i*spacing:(i+1)*width+i*spacing]=tmp[0:height,0:width]

    save_image(output,path+'output.png')


if __name__ == '__main__':
    main()
