#!/usr/bin/python

# Standard Libraries
import sys
import os
import pdb
import time
import math
import numpy
import Image
import scipy.ndimage.filters

# Non-Standard Libraries
from util import *

KERNEL_DIM = 5

def usage():
    # Sample Usage: python lucas_kanade_optical_flow.py a.jpg b.jpg 5 out.pfm
    print >> sys.stderr, 'python '+__file__+' image_a image_b spatial_sigma output_pfm'
    sys.exit(1)

def main():
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True)
    if len(sys.argv) < 5:
        usage()
    print 
    
    A = numpy.array(Image.open(sys.argv[1])).astype('float')
    B = numpy.array(Image.open(sys.argv[2])).astype('float')
    spatial_sigma = float(sys.argv[3])
    out_pfm_file_name = os.path.abspath(sys.argv[4])
    
    A_grayscale = convert_to_grayscale(A)
    B_grayscale = convert_to_grayscale(B)
    
    Ax = scipy.ndimage.filters.sobel(A_grayscale,axis=1)
    Ay = scipy.ndimage.filters.sobel(A_grayscale,axis=0)
    At = B_grayscale-A_grayscale
    
    gaussian_kernel = get_gaussian_kernel(KERNEL_DIM, spatial_sigma) # To use Gaussian weights
#    gaussian_kernel = numpy.ones([KERNEL_DIM,KERNEL_DIM], dtype='float') # To use no weights
    Ax_squared = numpy.square(Ax)
    Ay_squared = numpy.square(Ay)
    Ax_squared_blurred = convolve(Ax_squared, gaussian_kernel, zero_borders=True)
    Ay_squared_blurred = convolve(Ay_squared, gaussian_kernel, zero_borders=True)
    multiply_vectorized = numpy.vectorize(multiply)
    Axy = multiply_vectorized(Ax, Ay)
    Axy_blurred = convolve(Axy, gaussian_kernel, zero_borders=True)
    Axt = multiply_vectorized(Ax, At)
    Ayt = multiply_vectorized(Ay, At)
    Axt_blurred = convolve(Axt, gaussian_kernel, zero_borders=True)
    Ayt_blurred = convolve(Axt, gaussian_kernel, zero_borders=True)
    
    def calculate_velocity(weighted_sum_of_squared_x_derivative, weighted_sum_of_squared_y_derivative, weighted_sum_of_xy_derivative, weighted_sum_of_xt_derivative, weighted_sum_of_yt_derivative):
        ans = numpy.dot( 
                          numpy.linalg.pinv( numpy.array(
                                                          [ [weighted_sum_of_squared_x_derivative, weighted_sum_of_xy_derivative] ,
                                                            [weighted_sum_of_xy_derivative, weighted_sum_of_squared_y_derivative] ]
                                                        )
                                           )
                         ,
                         numpy.array([-weighted_sum_of_xt_derivative, -weighted_sum_of_yt_derivative])
                       )
        V_x = ans[0]
        V_y = ans[1]
        return V_x, V_y
    
    calculate_velocity_vectorized = numpy.vectorize(calculate_velocity)
    
    velocity_map = numpy.empty(A.shape, dtype='float')
    
    velocity_map[:,:,0],velocity_map[:,:,1] = calculate_velocity_vectorized(Ax_squared_blurred, Ay_squared_blurred, Axy_blurred, Axt_blurred, Ayt_blurred)
    
    save_image(numpy.square(velocity_map[:,:,0]),'V_x.png')
    save_image(numpy.square(velocity_map[:,:,1]),'V_y.png')
    
    writepfm(velocity_map, out_pfm_file_name)
    
    pdb.set_trace()

if __name__ == '__main__':
    main()

