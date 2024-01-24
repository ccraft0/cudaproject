build:
  nvcc -I/usr/include/opencv4 -lopencv_imgcodecs -lopencv_core -lnppif -lnppisu src/edgedetect.cpp -o bin/edgedetect.exe
