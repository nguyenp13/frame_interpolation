import sys
import os
import pdb
import time
import numpy as np
import Image
import random
import scipy.ndimage.filters

from util import *

padSize = 3000

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

def patch_distance_SSD_vectorized(A,B,patchRadius):
    distance = numpy.sum((A[:,:,:3]-B[:,:,:3])**2)
    #pdb.set_trace()
    return distance

def patchMatch(A, B, patchRadius, numIterations):
    A_padded = np.zeros((A.shape[0]+2*padSize,A.shape[1]+2*padSize,A.shape[2]))
    A_padded[padSize:A.shape[0]+padSize,padSize:A.shape[1]+padSize] = A[0:A.shape[0],0:A.shape[1]]
    B_padded = np.zeros((B.shape[0]+2*padSize,B.shape[1]+2*padSize,B.shape[2]))
    B_padded[padSize:B.shape[0]+padSize,padSize:B.shape[1]+padSize] = B[0:B.shape[0],0:B.shape[1]]
    
    A_padded = np.pad(A,((padSize,padSize),(padSize,padSize),(0,0)),'edge')
    B_padded = np.pad(B,((padSize,padSize),(padSize,padSize),(0,0)),'edge')
    
    offsetArray = initialize_patchmatch(A_padded, B_padded, patchRadius)
    #offsetArray = propagate_patchmatch(A,B,offsetArray,patchRadius,numIterations)
    offsetArray = propagate_patchmatch(A_padded,B_padded,offsetArray,patchRadius,numIterations)
    cpy_offsetArray = np.zeros((offsetArray.shape))
    for y in range(offsetArray.shape[0]):
        for x in range(offsetArray.shape[1]):
            cpy_offsetArray[y,x] = [offsetArray[y,x,1],offsetArray[y,x,0],offsetArray[y,x,2]]
    #pdb.set_trace()
    return offsetArray

def propagate_patchmatch(A, B, offsetArray, patchRadius, numIterations):
    print 'Propagating...'
    for i in range(2*numIterations):
        for yy in range(1,A.shape[0]-2*padSize):
            for xx in range(1,A.shape[1]-2*padSize):
                y = yy
                x = xx
                left_right = -1
                above_below = -1
                # if i is odd, scan from top_left to bottom_right, check the patches left and above to patch[y,x]
                # if i is even, scan from bottom_right to top_left, check the patches right and below to patch[y,x]
                if i%2 != 0:
                    y = A.shape[0]-2*padSize-1-yy
                    x = A.shape[1]-2*padSize-1-xx
                    left_right = 1
                    above_below = 1
                
                y_A = y+padSize
                x_A = x+padSize
                patchA = A[y_A-patchRadius:y_A+patchRadius+1, x_A-patchRadius:x_A+patchRadius+1, :]
                
                y_B = y_A+offsetArray[y,x+left_right,0]
                x_B = x_A+offsetArray[y,x+left_right,1]
                patchB = B[y_B-patchRadius:y_B+patchRadius+1, x_B-patchRadius:x_B+patchRadius+1, :]
                left_right_distance = patch_distance_SSD_vectorized(patchA,patchB,patchRadius)
                
                y_B = y_A+offsetArray[y+above_below,x,0]
                x_B = x_A+offsetArray[y+above_below,x,1]
                patchB = B[y_B-patchRadius:y_B+patchRadius+1, x_B-patchRadius:x_B+patchRadius+1, :]
                above_below_distance = patch_distance_SSD_vectorized(patchA,patchB,patchRadius)
                
                if( offsetArray[y,x,2] > left_right_distance ):
                    offsetArray[y,x] = [offsetArray[y,x+left_right,0],offsetArray[y,x+left_right,1],left_right_distance]
                if( offsetArray[y,x,2] > above_below_distance ):
                    offsetArray[y,x] = [offsetArray[y+above_below,x,0],offsetArray[y+above_below,x,1],above_below_distance]
    print 'Propagation has been done!'
    return offsetArray

def initialize_patchmatch( A_padded, B_padded, patchRadius ):
    print 'Initializing...'
    random.seed()
    offsetArray = np.zeros((A_padded.shape[0]-2*padSize,A_padded.shape[1]-2*padSize,3))
    for y in range(A_padded.shape[0]-2*padSize):
        for x in range(A_padded.shape[1]-2*padSize):
            offsetArray[y,x,0] = random.randint(0, A_padded.shape[0]-2*padSize-1) - y
            offsetArray[y,x,1] = random.randint(0, A_padded.shape[0]-2*padSize-1) - x
            yy = y+padSize
            xx = x+padSize
            patchA = A_padded[yy-patchRadius:yy+patchRadius+1, xx-patchRadius:xx+patchRadius+1, :]
            patchB = B_padded[yy+offsetArray[y,x,0]-patchRadius:yy+offsetArray[y,x,0]+patchRadius+1, xx+offsetArray[y,x,1]-patchRadius:xx+offsetArray[y,x,1]+patchRadius+1, :]
            offsetArray[y,x,2] = patch_distance_SSD_vectorized(patchA,patchB,patchRadius)
    print 'Initialization has been done!'
    return offsetArray

def main():
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True)
    if len(sys.argv) < 4:
        usage()
    print
    
    A = numpy.array(Image.open(sys.argv[1])).astype('float')
    B = numpy.array(Image.open(sys.argv[2])).astype('float')
    out_pfm_file_name = os.path.abspath(sys.argv[3])
    patchRadius = int(get_command_line_param_val(sys.argv, '-patchRadius', 'Error: patchRadius must be specified.', 'Error: Problem with specified patch radius.'))
    numIterations = int(get_command_line_param_val(sys.argv, '-numIterations', 'Error: numIterations of patchmatch must be specified.', 'Error: Problem with specified number of iterations.'))
    
    offsetArray = patchMatch(A,B,patchRadius,numIterations)
    visualize_optical_flow(offsetArray, out_pfm_file_name[0:len(out_pfm_file_name)-4]+'.png')
    correspondences = convert_velocity_map_to_absolute_coordinates(offsetArray)
    writepfm(correspondences,out_pfm_file_name)

if __name__ == '__main__':
    main()