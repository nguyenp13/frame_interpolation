#!/bin/bash

END=20
patchRadius=17
numIterations=6
numFrames=20
numPaddingFrames=0

dir1="../images/"
file="blocks"
tmp="/"
tmp2="_"
# name example: "../images/sphere/sphere"
name=$dir1$file$tmp$file

# dir_frames example: "../images/sphere/out/"
tmpName="out/"
dir_frames=$dir1$file$tmp$tmpName

mkdir $dir_pfm
mkdir $dir_frames

frameName="frame"
fileformatIMG=".png"
fileformatPFM=".pfm"

#for (( loop=$START; loop<=$END; loop++ ))
for loop in $(seq 1 $END);
do
    var=$((loop + 1))
    inputname1=$name$loop$fileformatIMG
    inputname2=$name$var$fileformatIMG
    outputname1=$name$tmp2$loop$var$fileformatPFM
    outputname2=$name$tmp2$var$loop$fileformatPFM
    outputFrameName=$dir_frames$frameName
    echo "inputname1 = $inputname1"
    echo "inputname2 = $inputname2"
    echo "outputname1 = $outputname1"
    echo "outputname2 = $outputname2"
    echo "outputFrameName = $outputFrameName"
    tmpPadding=$((loop - 1))
    numPaddingFrames=$(($numPaddingFrames*$tmpPadding))


    python ../edge_preserving_patchmatch_optical_flow.py $inputname1 $inputname2 $outputname1 -patchRadius $patchRadius -numIterations $numIterations
    python ../edge_preserving_patchmatch_optical_flow.py $inputname2 $inputname1 $outputname2 -patchRadius $patchRadius -numIterations $numIterations
    python ../warp.py $inputname1 $inputname2 $outputname1 $outputname2 $outputFrameName -num_frames $numFrames -num_padding_frames $numPaddingFrames
done
    #python edge_preserving_patchmatch_optical_flow_2.py ./images/sphere1.jpg ./images/sphere2.jpg ./images/sphere12.pfm -patchRadius 17 -numIterations 5