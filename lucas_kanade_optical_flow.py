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

KERNEL_DIM = 3

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
    
    A_grayscale = convert_to_grayscale(A)
    B_grayscale = convert_to_grayscale(B)
    
    Ax = scipy.ndimage.filters.sobel(A_grayscale,axis=1)
    Ay = scipy.ndimage.filters.sobel(A_grayscale,axis=0)
    At = B_grayscale-A_grayscale
    
#    gaussian_kernel = get_gaussian_kernel(KERNEL_DIM, spatial_sigma)
    gaussian_kernel = numpy.ones([KERNEL_DIM,KERNEL_DIM], dtype='float')
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
    
    Image.fromarray((numpy.square(V_x)).astype('uint8')).save('V_x.png')
    Image.fromarray((numpy.square(V_y)).astype('uint8')).save('V_y.png')
    Image.fromarray(50*numpy.sqrt(numpy.square(V_x)+numpy.square(V_y)).astype('uint8')).save('V_magnitude.png')
    
    print 'V_x', numpy.sum(V_x)
    print 'V_x', numpy.sum(V_y)
    print 'Ax', numpy.sum(Ax)
    print 'Ay', numpy.sum(Ay)
    print 'A', numpy.sum(A)
    print 'Ax', numpy.all(Ax==0)
    print 'Ay', numpy.all(Ay==0)
    print 'V_x', numpy.all(V_x==0)
    print 'V_y', numpy.all(V_y==0)
    print '1-'*88
    print 'Ax_squared_blurred', numpy.all(Ax_squared_blurred==0)
    print 'Ay_squared_blurred', numpy.all(Ay_squared_blurred==0)
    print '2-'*88
    print 'Axy_blurred', numpy.all(Axy_blurred==0)
    print 'At', numpy.all(At==0)
    print 'Axt', numpy.all(Axt==0)
    print 'Ayt', numpy.all(Ayt==0)
    print '3-'*88
    print 'Axt_blurred', numpy.all(Axt_blurred==0)
    print 'Ayt_blurred', numpy.all(Ayt_blurred==0)
    print 'Ayt_blurred', numpy.all(Ayt_blurred==0)
    print 
    print 'V_x'
    print V_x
    print 
    print 'V_y'
    print V_y
    print 
    pdb.set_trace()

if __name__ == '__main__':
    main()

