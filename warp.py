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
    # Sample Usage: python warp.py 1.png 2.png ann.pfm bnn.pfm ./out/frame -num_frames 5 -num_padding_frames 0
    print >> sys.stderr, 'python '+__file__+' image_a image_b ann_correspondence_pfm bnn_correspondence_pfm output_prefix'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options: -num_frames <int_val>'
    print >> sys.stderr, '         -num_padding_frames <int_val>'
    print >> sys.stderr, ''
    sys.exit(1)

def main():
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True)
    if len(sys.argv) < 6:
        usage()
    print 
    
    A0 = numpy.array(Image.open(sys.argv[1])).astype('float')
    B0 = numpy.array(Image.open(sys.argv[2])).astype('float')
    ann_pfm_file_name = os.path.abspath(sys.argv[3])
    bnn_pfm_file_name = os.path.abspath(sys.argv[4])
    output_prefix = os.path.abspath(sys.argv[5])
    num_frames = int(get_command_line_param_val(sys.argv, '-num_frames', 'Error: Kernel width must be specified.', 'Error: Problem with specified kernel width.'))
    num_padding_frames = int(get_command_line_param_val(sys.argv, '-num_padding_frames', 'Error: Number of padding frames must be specified.', 'Error: Problem with specified number of padding.'))
    
    print "Parameters: "
    print "    num_frames: %s" % str(num_frames)
    print 
    
    ann = readpfm(ann_pfm_file_name)
    bnn = readpfm(bnn_pfm_file_name)
    assert ann.shape == bnn.shape
    h, w = ann.shape[:2]
    A = A0[:h,:w,:]
    B = B0[:h,:w,:]
    
    out = numpy.empty(A.shape, dtype='uint8')
    
    for i,t in enumerate([e / float(num_frames-1) for e in xrange(num_frames)]):
        A_to_B_warp = round_vectorized(warp(A, bnn, t))
        B_to_A_warp = round_vectorized(warp(B, ann, 1-t))
        
        out = cross_dissolve_vectorized(A_to_B_warp, B_to_A_warp, t)
        
        save_image(out,output_prefix+'_'+str(num_padding_frames+i)+'.png')
    for i in xrange(num_padding_frames):
        save_image(A,output_prefix+'_'+str(i)+'.png')
        save_image(B,output_prefix+'_'+str(num_padding_frames+num_frames+i)+'.png')
        

if __name__ == '__main__':
    main()

