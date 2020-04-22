cd build/Release
mkdir $myDIR
$myDIR="LOGS"
For ($i=6; $i -ge 0; $i--) {
    $VAR=[math]::Pow(10, $i)
    ./Acoustic_Raytracing.out.exe $VAR > $myDIR/$VAR.log
}