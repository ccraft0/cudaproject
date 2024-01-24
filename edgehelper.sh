# A simple script for running edgedetect on all files in a folder
ls *.jpg > edgehelperaux.txt
ls *.png >> edgehelperaux.txt
ls *.bmp >> edgehelperaux.txt
ls *.tiff >> edgehelperaux.txt
./edgedetect.exe edgehelperaux.txt "$@"
rm edgehelperaux.txt