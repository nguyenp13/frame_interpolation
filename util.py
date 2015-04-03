
# Common functions used for the programming assignment

import sys
import os
import ntpath
import math
import numpy
import Image
import scipy.ndimage.filters
import matplotlib
import pdb

inf = float('inf')
PI = math.pi

X_COORD = 1
Y_COORD = 0

round_vectorized = numpy.vectorize(round)
atan2_vectorized = numpy.vectorize(math.atan2)

def save_image(image, name): 
    final_output = clamp_array(image) 
    final_output = final_output.astype('uint8') 
    Image.fromarray(final_output).save(name) 

def convert_velocity_map_to_absolute_coordinates(velocity_map):
    height, width = velocity_map.shape[:2]
    coords_map = numpy.empty([height, width, 3], dtype='float')
    coords_map[:,:,X_COORD] = numpy.tile(numpy.arange(width),[height,1])
    coords_map[:,:,Y_COORD] = numpy.tile(numpy.arange(height)[:,None],[1,width])
    return round_vectorized(coords_map+velocity_map)

def visualize_optical_flow(velocity_map, output_file=None):
    velocity_map_magnitudes = numpy.sqrt(numpy.square(velocity_map[:,:,X_COORD])+numpy.square(velocity_map[:,:,Y_COORD]))
    velocity_map_magnitudes /= numpy.max(velocity_map_magnitudes,axis=None) # Normalize the vectors
    angle_map = (atan2_vectorized(velocity_map[:,:,Y_COORD], velocity_map[:,:,X_COORD])*180/PI)%360 # in degrees, not radians
    
    visualization_hsv = numpy.empty(velocity_map.shape, dtype='float')
    visualization_hsv[:,:,0] = angle_map / 360.0 # hue
    visualization_hsv[:,:,1] = velocity_map_magnitudes # saturation
    visualization_hsv[:,:,2] = 1 # value
    visualization_rgb = (matplotlib.colors.hsv_to_rgb(visualization_hsv)*255).astype('uint8')
    if output_file is not None:
        save_image(visualization_rgb, output_file)
    return visualization_rgb, velocity_map_magnitudes, angle_map

def visualize_optical_flow_with_arrows(velocity_map, output_file=None, box_width = 50):
    visualization_image, magnitude_map, angle_map = visualize_optical_flow(velocity_map, None)
    h,w = visualization_image.shape[:2]
    def line(x0,y0,x1,y1):
        distance = int(round(math.sqrt((x1-x0)**2+(y1-y0)**2)))
        for i in xrange(distance):
            t = float(i)/distance
            x = x0*(1-t)+x1*t
            y = y0*(1-t)+y1*t
            visualization_image[y,x,:] = [0,0,0]
            for yy in [y-1,y,y+1]:
                for xx in [x-1,x,x+1]:
                    if yy >= 0 and yy < h and xx >= 0 and xx < w:
                        visualization_image[yy,xx] /= 2.0
    for y in xrange(0, h, box_width):
        for x in xrange(0, w, box_width):
            if x+box_width<w and y+box_width<h:
                center_y = round(y+box_width/2.0)
                center_x = round(x+box_width/2.0)
                line_length = math.sqrt(numpy.mean(magnitude_map[y:y+box_width,x:x+box_width]))*box_width/2.0
                angle_degrees = numpy.mean(angle_map[y:y+box_width,x:x+box_width],axis=None)
                angle_radians = angle_degrees*math.pi/180.0
                
                dx_rel = math.cos(angle_radians)
                dy_rel = -math.sin(angle_radians) # negative because the vertical axis in image coordinates is the opposite of what it is in Cartesian coordinates
                end_point_x = center_x+dx_rel*line_length
                end_point_y = center_y+dy_rel*line_length
                line(center_x, center_y, end_point_x, end_point_y)
                left_wing_angle_degrees = (angle_degrees+135)%360
                left_wing_angle_radians = left_wing_angle_degrees*math.pi/180.0
                left_wing_dx_rel = math.cos(left_wing_angle_radians)
                left_wing_dy_rel = -math.sin(left_wing_angle_radians)
                left_wing_x = end_point_x+left_wing_dx_rel*line_length/2.0
                left_wing_y = end_point_y+left_wing_dy_rel*line_length/2.0
                line(end_point_x, end_point_y, left_wing_x, left_wing_y)
                right_wing_angle_degrees = (angle_degrees-135)%360
                right_wing_angle_radians = right_wing_angle_degrees*math.pi/180.0
                right_wing_dx_rel = math.cos(right_wing_angle_radians)
                right_wing_dy_rel = -math.sin(right_wing_angle_radians)
                right_wing_x = end_point_x+right_wing_dx_rel*line_length/2.0
                right_wing_y = end_point_y+right_wing_dy_rel*line_length/2.0
                line(end_point_x, end_point_y, right_wing_x, right_wing_y)
    gaussian_kernel = get_gaussian_kernel(13, 2)
    save_image(visualization_image,output_file)
    return visualization_image, magnitude_map, angle_map

def generate_optical_flow_visualization_legend(output_file=None, width=1001, height=1001):
    center_x = width/2
    center_y = height/2
    coords_map = numpy.empty([height, width, 3], dtype='float')
    coords_map[:,:,X_COORD] = numpy.tile(numpy.arange(width),[height,1])-center_x
    coords_map[:,:,Y_COORD] = numpy.tile(numpy.arange(height-1,-1,-1)[:,None],[1,width])-center_y
    return visualize_optical_flow(coords_map, output_file)

def generate_optical_flow_visualization_legend_with_arrows(output_file=None, width=1001, height=1001, box_width = 50):
    center_x = width/2
    center_y = height/2
    coords_map = numpy.empty([height, width, 3], dtype='float')
    coords_map[:,:,X_COORD] = numpy.tile(numpy.arange(width),[height,1])-center_x
    coords_map[:,:,Y_COORD] = numpy.tile(numpy.arange(height-1,-1,-1)[:,None],[1,width])-center_y
    return visualize_optical_flow_with_arrows(coords_map, output_file, box_width=box_width)

def assertion(condition, message):
    if not condition:
        print >> sys.stderr, ''
        print >> sys.stderr, message
        print >> sys.stderr, ''
        sys.exit(1)

def get_command_line_param_val(args, param_option, param_option_not_specified_error_message, param_val_not_specified_error_message):
    assertion(param_option in args, param_option_not_specified_error_message)
    param_val_index = 1+args.index(param_option)
    assertion(param_val_index < len(args), param_val_not_specified_error_message)
    return args[param_val_index]

def readpfm(filename):
    f = open(filename, 'rb')
    line = f.readline()
    assert len(line) >= 2
    assert line[0] == 'P', line
    if line[1] == 'f':
        channels = 1
    elif line[1] == 'F':
        channels = 3
    else:
        raise ValueError('bad format')
    (w, h) = f.readline().split()
    w = int(w)
    h = int(h)
    scale = float(f.readline())
    byteorder = '<' if scale<0 else '>'
    dtype = byteorder + 'f4'
    I = numpy.fromfile(f, dtype, w*h*channels)
    f.close()
    return numpy.flipud(I.reshape((h, w, channels))).astype('float')

def writepfm(I, filename):
    if len(I.shape) == 2:
        I = numpy.dstack((I,))
    (h, w, channels) = I.shape
    with open(filename, 'wb') as f:
        byteorder = '>'
        assert channels in [1, 3]
        f.write('P' + ('f' if channels == 1 else 'F') + '\n')
        f.write('%d %d\n' % (w, h))
        f.write('1.0\n')
        I = numpy.asarray(I, byteorder + 'f4')
        f.write(numpy.flipud(I).tostring())

def pfm_to_png(filename):
    I = readpfm(filename)
    rescaled = (255.0 * I).astype(numpy.uint8)
    im = Image.fromarray(rescaled)
    im.save(filename[:-4]+'.png')

def system(cmd):
    pass
    print cmd
    os.system(cmd)

def list_from_set(s):
    return [ e for e in s ]

def mean(l):
    return sum(l) / float(len(l))

def list_dir_abs(basepath):
    return map(lambda x: os.path.abspath(os.path.join(basepath, x)), os.listdir(basepath))

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_containing_folder(path):
    head, tail = ntpath.split(path)
    return head

def makedirs_recursive(dirname):
    if not os.path.exists(dirname):
        makedirs_recursive(get_containing_folder(dirname))
        os.makedirs(dirname)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def G(x, sigma):
    return math.exp(-(x*x)/(2*(sigma*sigma))) / sqrt(2*PI*(sigma*sigma));

def G2(x, y,sigma):
    return math.exp(-((y*y)+(x*x))/(2*(sigma*sigma))) / (2*PI*(sigma*sigma));

def divide(top, bottom):
    if bottom == 0.0:
        return inf
    return top / bottom

def multiply(a, b):
    return a*b
multiply_vectorized = numpy.vectorize(multiply)

def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def radians_to_degrees(radian_value):
    return radian_value*180/PI

def downsample_2d(I0, downsample_factor=2):
    I = numpy.array(I0,dtype='float')
    return I[::2,::2]

def convert_to_grayscale(I0):
    I = numpy.array(I0,dtype='float')
    I[:,:,0] *= 0.299
    I[:,:,1] *= 0.5870
    I[:,:,2] *= 0.1140
    I_grayscale = numpy.mean(I, axis=2)
    return I_grayscale

def convolve(I0, k, zero_borders=False): # Convolves RGB image with 2D kernel 
    assert len(k.shape) == 2, "Kernel must be 2D"
    k_h, k_w = k.shape
    assert k_h % 2 == 1, "Kernel must have odd height"
    assert k_w % 2 == 1, "Kernel must have odd width"
    
    I = numpy.zeros(I0.shape)
    if len(I0.shape)==2:
        I_h, I_w = I.shape
        if zero_borders:
            I = scipy.ndimage.filters.convolve(I0, k, mode='constant', cval=0.0)
        else:
            I = scipy.ndimage.filters.convolve(I0, k, mode='nearest')
    else:
        I_h, I_w, I_channels = I0.shape
        for channel in xrange(I_channels):
            if zero_borders:
                I[:,:,channel] = scipy.ndimage.filters.convolve(I0[:,:,channel], k, mode='constant', cval=0.0)
            else:
                I[:,:,channel] = scipy.ndimage.filters.convolve(I0[:,:,channel], k, mode='nearest')
#    import pdb; pdb.set_trace()
    return I

def normalize(array0):
    assert len(array0.shape) in [2,3], "normalize() is only supported for 2D and 3D arrays"
    array = None
    if len(array0.shape) == 2:
        array_sum = numpy.sum(array0)
        array = numpy.array(array0)
        h, w = array0.shape
        for y in xrange(h):
            for x in xrange(w):
                array[y,x] /= array_sum
    elif len(array0.shape) == 3:
        array = numpy.array(array0)
        h, w, channels = array0.shape
        for z in xrange(channels):
            array_sum = numpy.sum(array0[:,:,z])
            for y in xrange(h):
                for x in xrange(w):
                    array[y,x,z] /= array_sum
    assert array is not None
    return array

def clamp(x, min_val=0, max_val=255):
    return max( min(x, max_val), min_val)

clamp_array = numpy.vectorize(clamp)

def get_gaussian_kernel(dim, sigma):
    assert dim % 2==1, "Gaussian kernel must be of odd dimension"
    
    kernel = numpy.zeros(shape=(dim,dim))
    
    for y in xrange(dim):
        for x in xrange(y+1):
            weight = G2(y-dim/2, x-dim/2, sigma)
            kernel[y,x] = weight
            if x != y:
                kernel[x,y] = kernel[y,x]
    kernel = normalize(kernel)
    return kernel

