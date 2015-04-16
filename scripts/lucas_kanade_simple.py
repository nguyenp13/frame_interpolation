#!/usr/bin/python

# Standard Libraries
import sys; sys.path += ['..'] 
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
    # Sample Usage: python lucas_kanade_simple.py ../1.png ../2.png ./results -num_frames 50
    print >> sys.stderr, 'python '+__file__+' image_a image_b output_directory' 
    print >> sys.stderr, '' 
    sys.exit(1) 

def main(): 
    numpy.set_printoptions(linewidth=200, precision=5, suppress=True) 
    if len(sys.argv) < 4: 
        usage() 
    print 
    
    ann_pfm_file_name = os.path.abspath(generate_random_file_name('pfm'))
    bnn_pfm_file_name = os.path.abspath(generate_random_file_name('pfm'))
    
    try:
        image_a_file_name = os.path.abspath(sys.argv[1])
        image_b_file_name = os.path.abspath(sys.argv[2])
        output_directory = os.path.abspath(sys.argv[3])
        spatial_sigma = float(get_command_line_param_val_default_value(sys.argv,'-spatial_sigma','5'))
        kernel_dim = int(get_command_line_param_val_default_value(sys.argv,'-kernel_dim','31'))
        num_frames = int(get_command_line_param_val_default_value(sys.argv,'-num_frames','5'))
        num_padding_frames = int(get_command_line_param_val_default_value(sys.argv,'-num_padding_frames','0'))
        
        print "Parameters: " 
        print "    image_a_file_name: %s" % str(image_a_file_name) 
        print "    image_b_file_name: %s" % str(image_b_file_name) 
        print "    output_directory: %s" % str(output_directory) 
        print 
        
        makedirs(output_directory)
        os.system('((python ../lucas_kanade_optical_flow.py '+image_a_file_name+' '+image_b_file_name+' '+ann_pfm_file_name+' -spatial_sigma '+str(spatial_sigma)+' -kernel_dim '+str(kernel_dim)+' -num_iterations 1) & (python ../lucas_kanade_optical_flow.py '+image_b_file_name+' '+image_a_file_name+' '+bnn_pfm_file_name+' -spatial_sigma '+str(spatial_sigma)+' -kernel_dim '+str(kernel_dim)+' -num_iterations 1) & wait && (python ../warp.py '+image_a_file_name+' '+image_b_file_name+' '+ann_pfm_file_name+' '+bnn_pfm_file_name+' '+os.path.join(output_directory,'frame')+' -num_frames '+str(num_frames)+' -num_padding_frames '+str(num_padding_frames)+')) > /dev/null 2>&1')
    finally:
        os.system('rm '+ann_pfm_file_name+' > /dev/null 2>&1')
        os.system('rm '+bnn_pfm_file_name+' > /dev/null 2>&1')
    print "Done."
    print 

if __name__ == '__main__':
    main()

