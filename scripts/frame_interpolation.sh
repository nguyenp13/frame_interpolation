#!/bin/bash

END=18

patchRadius=50
numIterations=6
numFrames=5
numPaddingFrames=0

spacialSigma=5
kernelDim=31

dir1="../images/"
file="vcbox"
tmp="/"
tmp_patchMatch="_patchMatch_"
tmp_lucasKanada="_lucasKanada_"
# name example: "../images/sphere/sphere"
name=$dir1$file$tmp$file

# dir_frames example: "../images/sphere/out/"
tmpName="out/"
tmpName_patchmatch="patchMatch/"
tmpName_lucasKanada="lucasKanada/"
dir_frames=$dir1$file$tmp$tmpName
outputFrameNamePatchMatch=$dir_frames$tmpName_patchmatch
outputFrameNameLucasKanada=$dir_frames$tmpName_lucasKanada

#mkdir $dir_pfm
mkdir $dir_frames
mkdir $outputFrameNamePatchMatch
mkdir $outputFrameNameLucasKanada

frameName="frame"
fileformatIMG=".png"
fileformatPFM=".pfm"

outputFrameNamePatchMatch=$outputFrameNamePatchMatch$frameName
outputFrameNameLucasKanada=$outputFrameNameLucasKanada$frameName

#for (( loop=$START; loop<=$END; loop++ ))
for loop in $(seq 1 $END);
do
    var=$((loop + 1))
    inputname1=$name$loop$fileformatIMG
    inputname2=$name$var$fileformatIMG
    outputname_patchMatch1=$name$tmp_patchMatch$loop$var$fileformatPFM
    outputname_patchMatch2=$name$tmp_patchMatch$var$loop$fileformatPFM
    outputname_lucasKanada1=$name$tmp_lucasKanada$loop$var$fileformatPFM
    outputname_lucasKanada2=$name$tmp_lucasKanada$var$loop$fileformatPFM

    echo "inputname1 = $inputname1"
    echo "inputname2 = $inputname2"
    echo "outputname_patchMatch1 = $outputname_patchMatch1"
    echo "outputname_patchMatch2 = $outputname_patchMatch2"
    echo "outputname_lucasKanada1 = $outputname_lucasKanada1"
    echo "outputname_lucasKanada2 = $outputname_lucasKanada2"
    echo "outputFrameNamePatchMatch = $outputFrameNamePatchMatch"
    echo "outputFrameNameLucasKanada = $outputFrameNameLucasKanada"
    tmpPadding=$((loop - 1))
    numPaddingFrames=$(($numFrames * $tmpPadding))

    python ../lucas_kanade_optical_flow.py $inputname1 $inputname2 $outputname_lucasKanada1 -spatial_sigma $spacialSigma -kernel_dim $kernelDim -num_iterations $numIterations
    python ../lucas_kanade_optical_flow.py $inputname2 $inputname1 $outputname_lucasKanada2 -spatial_sigma $spacialSigma -kernel_dim $kernelDim -num_iterations $numIterations
    python ../warp.py $inputname1 $inputname2 $outputname_lucasKanada1 $outputname_lucasKanada2 $outputFrameNameLucasKanada -num_frames $numFrames -num_padding_frames $numPaddingFrames


    python ../edge_preserving_patchmatch_optical_flow.py $inputname1 $inputname2 $outputname_patchMatch1 -patchRadius $patchRadius -numIterations $numIterations
    python ../edge_preserving_patchmatch_optical_flow.py $inputname2 $inputname1 $outputname_patchMatch2 -patchRadius $patchRadius -numIterations $numIterations
    python ../warp.py $inputname1 $inputname2 $outputname_patchMatch1 $outputname_patchMatch2 $outputFrameNamePatchMatch -num_frames $numFrames -num_padding_frames $numPaddingFrames
done
    #python edge_preserving_patchmatch_optical_flow_2.py ./images/sphere1.jpg ./images/sphere2.jpg ./images/sphere12.pfm -patchRadius 17 -numIterations 5