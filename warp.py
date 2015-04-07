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
    # Sample Usage: python warp.py 1.png 2.png ann.pfm bnn.pfm -num_frames 5 
    print >> sys.stderr, 'python '+__file__+' image_a image_b ann_correspondence_pfm bnn_correspondence_pfm output_prefix'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options: -num_frames <int_val>'
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
    
    print "Parameters: "
    print "    num_frames: %s" % str(num_frames)
    print 
    
    ann = readpfm(ann_pfm_file_name)
    bnn = readpfm(bnn_pfm_file_name)
    assert ann.shape == bnn.shape
    h, w = ann.shape[:2]
    A = A0[:h,:w,:]
    B = B0[:h,:w,:]
    
    out = numpy.zeros(A.shape, dtype='uint8')
    
    for i,t in enumerate([e / float(num_frames-1) for e in xrange(num_frames)]):
        for y in xrange(h):
            for x in xrange(w):
                a_warp_source_color_x = int(round(lerp(x, bnn[y,x,X_COORD], t)))
                a_warp_source_color_y = int(round(lerp(y, bnn[y,x,Y_COORD], t)))
                b_warp_source_color_x = int(round(lerp(x, ann[y,x,X_COORD], 1-t)))
                b_warp_source_color_y = int(round(lerp(y, ann[y,x,Y_COORD], 1-t)))
                
                cross_dissolved_red =   round(lerp(A[a_warp_source_color_y, a_warp_source_color_x, R_COORD], B[b_warp_source_color_y, b_warp_source_color_x, R_COORD], t))
                cross_dissolved_green = round(lerp(A[a_warp_source_color_y, a_warp_source_color_x, G_COORD], B[b_warp_source_color_y, b_warp_source_color_x, G_COORD], t))
                cross_dissolved_blue =  round(lerp(A[a_warp_source_color_y, a_warp_source_color_x, B_COORD], B[b_warp_source_color_y, b_warp_source_color_x, B_COORD], t))
                
                out[y,x,R_COORD] = cross_dissolved_red
                out[y,x,G_COORD] = cross_dissolved_green
                out[y,x,B_COORD] = cross_dissolved_blue
        save_image(out,output_prefix+'_'+str(i)+'.png')

if __name__ == '__main__':
    main()

