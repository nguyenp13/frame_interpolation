import sys
import os
import pdb
import time
import numpy as np
import Image
import random
import scipy.ndimage.filters

from util import *

def usage():
    # Sample Usage: python lucas_kanade_optical_flow.py a.png b.png out.pfm -spatial_sigma 5 -kernel_dim 31 -num_iterations 1
    print >> sys.stderr, 'python '+__file__+' image_a image_b output_pfm'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options: -patchRadius <int_val>'
    print >> sys.stderr, '         -numIterations <int_val>'
    print >> sys.stderr, ''
    sys.exit(1)

def patch_distance_bilateral_weight(A,B,y_A,x_A,y_B,x_B,patchRadius):
    distance = 0.0
    for yy in range(-patchRadius, patchRadius+1):
        for xx in range(-patchRadius, patchRadius+1):
            y = yy
            x = xx
            y_A_clampled == y_A+yy
            if y_A_clampled < 0:
                weight = 0
            distance += 0.1
    return distance

def patch_distance_SSD(A,B,y_A,x_A,y_B,x_B,patchRadius):
    distance = 0.0
    for y in range(-patchRadius, patchRadius+1):
        for x in range(-patchRadius, patchRadius+1):
            y_A_clampled = y_A+y
            if y_A_clampled < 0:
                y_A_clampled = 0
            if y_A_clampled >= A.shape[0]:
                y_A_clampled = A.shape[0]-1
            x_A_clampled = x_A+x
            if x_A_clampled < 0:
                x_A_clampled = 0
            if x_A_clampled >= A.shape[0]:
                x_A_clampled = A.shape[1]-1
            
            y_B_clampled = y_B+y
            if y_B_clampled < 0:
                y_B_clampled = 0
            if y_B_clampled >= B.shape[0]:
                y_B_clampled = B.shape[0]-1
            x_B_clampled = x_B+x
            if x_B_clampled < 0:
                x_B_clampled = 0
            if x_B_clampled >= B.shape[0]:
                x_B_clampled = B.shape[1]-1
            
            for c in range(3):
                distance += math.pow( (A[y_A_clampled,x_A_clampled,c]-B[y_B_clampled,x_B_clampled,c]),2 )
    return distance

def patchMatch(A, B, patchRadius = 3, numIterations = 2):
    offsetArray = initialize_patchmatch(A, B, patchRadius)
    offsetArray = propagate_patchmatch(A,B,offsetArray,patchRadius,numIterations)
    cpy_offsetArray = np.zeros((offsetArray.shape))
    for y in range(offsetArray.shape[0]):
        for x in range(offsetArray.shape[1]):
            cpy_offsetArray[y,x] = [offsetArray[y,x,1],offsetArray[y,x,0],offsetArray[y,x,2]]
    #pdb.set_trace()
    return offsetArray

def propagate_patchmatch(A, B, offsetArray, patchRadius, numIterations):
    for i in range(2*numIterations):
        print 'Propagation iterations ', i/2
        for yy in range(1,A.shape[0]):
            for xx in range(1,A.shape[1]):
                y = yy
                x = xx
                left_right = -1
                above_below = -1
                # if i is odd, scan from top_left to bottom_right, check the patches left and above to patch[y,x]
                # if i is even, scan from bottom_right to top_left, check the patches right and below to patch[y,x]
                if i%2 != 0:
                    y = A.shape[0]-1-yy
                    x = A.shape[1]-1-xx
                    left_right = 1
                    above_below = 1
                left_right_distance = patch_distance_SSD(A,B,y,x,y+offsetArray[y,x+left_right,0],x+offsetArray[y,x+left_right,1],patchRadius)
                above_below_distance = patch_distance_SSD(A,B,y,x,y+offsetArray[y+above_below,x,0],x+offsetArray[y+above_below,x,1],patchRadius)
                if( offsetArray[y,x,2] > left_right_distance ):
                    offsetArray[y,x] = [offsetArray[y,x+left_right,0],offsetArray[y,x+left_right,1],left_right_distance]
                if( offsetArray[y,x,2] > above_below_distance ):
                    offsetArray[y,x] = [offsetArray[y+above_below,x,0],offsetArray[y+above_below,x,1],above_below_distance]

    return offsetArray

def initialize_patchmatch( A, B, patchRadius ):
    random.seed()
    offsetArray = np.zeros((A.shape[0],A.shape[1],3))
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            offsetArray[y,x,0] =random.randint(0, B.shape[0]-1) - y
            offsetArray[y,x,1] =random.randint(0, B.shape[1]-1) - x
            offsetArray[y,x,2] = patch_distance_SSD(A,B,y,x,y+offsetArray[y,x,0],x+offsetArray[y,x,1],patchRadius)
    return offsetArray

def main():
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True)
    if len(sys.argv) < 4:
        usage()
    print

    A = numpy.array(Image.open(sys.argv[1])).astype('float')
    B = numpy.array(Image.open(sys.argv[2])).astype('float')
    out_pfm_file_name = os.path.abspath(sys.argv[3])
    patchRadius = float(get_command_line_param_val(sys.argv, '-patchRadius', 'Error: patchRadius must be specified.', 'Error: Problem with specified patch radius.'))
    numIterations = int(get_command_line_param_val(sys.argv, '-numIterations', 'Error: numIterations of patchmatch must be specified.', 'Error: Problem with specified number of iterations.'))
    
    offsetArray = patchMatch(A,B)
    visualize_optical_flow(offsetArray, "output.png")
    correspondences = convert_velocity_map_to_absolute_coordinates(offsetArray)
    writepfm(correspondences,out_pfm_file_name)

if __name__ == '__main__':
    main()