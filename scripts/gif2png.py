from PIL import Image

def convertGifToPng(name):
    img = Image.open(name)
    im = img.convert('RGB')
    im.save(name[0:len(name)-4]+'.png', 'PNG')

def main():
    dir = '../images/blocks/blocks'
    for i in range(20):
        filename = dir+str(i)+'.gif'
        print filename
        convertGifToPng(filename)

if __name__ == "__main__": main()