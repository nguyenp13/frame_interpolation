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

def usage():
    # Sample Usage: python lucas_kanade_optical_flow.py a.png b.png out.pfm -spatial_sigma 5 -kernel_dim 31 -num_iterations 1
    print >> sys.stderr, 'python '+__file__+' image_a image_b output_pfm'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options: -spatial_sigma <float_val>'
    print >> sys.stderr, '         -kernel_dim <int_val>'
    print >> sys.stderr, '         -num_iterations <int_val>'
    print >> sys.stderr, ''
    sys.exit(1)

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
    
    return V_y, V_x

calculate_velocity_vectorized = numpy.vectorize(calculate_velocity)

def lucas_kanade_simple(A_grayscale, B_grayscale, gaussian_kernel):
    Ax = scipy.ndimage.filters.sobel(A_grayscale,axis=1)
    Ay = scipy.ndimage.filters.sobel(A_grayscale,axis=0)
    At = B_grayscale-A_grayscale
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
    
    velocity_map = numpy.empty(A_grayscale.shape+(3,), dtype='float')
    velocity_map[:,:,Y_COORD],velocity_map[:,:,X_COORD] = calculate_velocity_vectorized(Ax_squared_blurred, Ay_squared_blurred, Axy_blurred, Axt_blurred, Ayt_blurred)
    
#    print 'X_COORD'
#    print velocity_map[:,:,X_COORD]
#    print 'Y_COORD'
#    print velocity_map[:,:,Y_COORD]
#    print 
#    pdb.set_trace()
    
    return velocity_map

def main():
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True)
    if len(sys.argv) < 4:
        usage()
    print 
    
    A = numpy.array(Image.open(sys.argv[1])).astype('float')
    B = numpy.array(Image.open(sys.argv[2])).astype('float')
    out_pfm_file_name = os.path.abspath(sys.argv[3])
    spatial_sigma = float(get_command_line_param_val(sys.argv, '-spatial_sigma', 'Error: A spatial sigma must be specified.', 'Error: Problem with specified spatial sigma value.'))
    kernel_dim = int(get_command_line_param_val(sys.argv, '-kernel_dim', 'Error: Kernel width must be specified.', 'Error: Problem with specified kernel width.'))
    num_iterations = int(get_command_line_param_val(sys.argv, '-num_iterations', 'Error: Number of iterations must be specified.', 'Error: Problem with specified number of iterations.'))
    
    print "Parameters: "
    print "    spatial_sigma: %s" % str(spatial_sigma)
    print "    kernel_dim: %s" % str(kernel_dim)
    print "    num_iterations: %s" % str(num_iterations)
    print 
    
    A_grayscale = convert_to_grayscale(A)
    B_grayscale = convert_to_grayscale(B)
    
    gaussian_kernel = get_gaussian_kernel(kernel_dim, spatial_sigma) # To use Gaussian weights
#    gaussian_kernel = numpy.ones([kernel_dim,kernel_dim], dtype='float') # To use no weights
    velocity_map = lucas_kanade_simple(A_grayscale, B_grayscale, gaussian_kernel)
    
#    save_image(numpy.square(velocity_map[:,:,X_COORD]),'V_x.png')
#    save_image(numpy.square(velocity_map[:,:,Y_COORD]),'V_y.png')
    
    correspondences = convert_velocity_map_to_absolute_coordinates(velocity_map)
#    print 'X'
#    print correspondences[:,:,X_COORD]
#    print 'Y'
#    print correspondences[:,:,Y_COORD]
    writepfm(correspondences, out_pfm_file_name)
    
#    print "Done.", correspondences.shape, numpy.max(correspondences[:,:,Y_COORD], axis=None), numpy.max(correspondences[:,:,X_COORD], axis=None)
#    pdb.set_trace()

if __name__ == '__main__':
    main()

