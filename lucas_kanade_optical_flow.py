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

KERNEL_DIM = 15
BOX_THICKNESS = 2
BOX_COLOR = [0,0,255]

def get_smaller_eigenvalue(F_x_squared_val, F_y_squared_val, F_xy_val):
    eigen_values, eigen_vectors = numpy.linalg.eig(numpy.array(
                                                                [[F_x_squared_val,F_xy_val],
                                                                 [F_xy_val,F_y_squared_val]] 
                                                              ))
    return min(eigen_values)

def usage():
    # Sample Usage: python lucas_kanade_optical_flow.py a.jpg b.jpg out.pfm
    print >> sys.stderr, 'python '+__file__+' image_a image_b spatial_sigma output_pfm'
    sys.exit(1)

def main():
    if len(sys.argv) < 5:
        usage()
    print 
    
    A = numpy.array(Image.open(sys.argv[1])).astype('float')
    B = numpy.array(Image.open(sys.argv[2])).astype('float')
    spatial_sigma = float(sys.argv[3])
    
    A_grayscale = convert_to_grayscale(A)
    B_grayscale = convert_to_grayscale(B)
    
    sobel_x = numpy.array([[-1, 0, 1], # these are the sobel filter kernels
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype='float')
    sobel_y = numpy.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype='float')
    
    Ax = convolve(A_grayscale, sobel_x) # convolving with the sobel filter kernels
    Ay = convolve(A_grayscale, sobel_y) 
    At = B_grayscale-A_grayscale
    
    gaussian_kernel = get_gaussian_kernel(KERNEL_DIM, spatial_sigma)
    Ax_squared = numpy.square(Ax)
    Ay_squared = numpy.square(Ay)
    Ax_squared_blurred = convolve(Ax_squared, gaussian_kernel)
    Ay_squared_blurred = convolve(Ay_squared, gaussian_kernel)
    multiply_vectorized = numpy.vectorize(multiply)
    Axy = multiply_vectorized(Ax, Ay)
    Axy_blurred = convolve(Axy, gaussian_kernel)
    Axt = multiply_vectorized(Ax, At)
    Ayt = multiply_vectorized(Ay, At)
    Axt_blurred = convolve(Axt, gaussian_kernel)
    Ayt_blurred = convolve(Axt, gaussian_kernel)
    
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
    
    V_x, V_y = calculate_velocity_vectorized(Ax_squared_blurred, Ay_squared_blurred, Axy_blurred, Axt_blurred, Ayt_blurred)
    
    Image.fromarray(50*numpy.square(V_x).astype('uint8')).save('V_x.png')
    Image.fromarray(50*numpy.square(V_y).astype('uint8')).save('V_y.png')
    
#    pdb.set_trace()

if __name__ == '__main__':
    main()

