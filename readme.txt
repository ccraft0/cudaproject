Description: Takes an input file and implements edge detection using one of four filters, either
Scharr, Sobel, Prewitt, or Roberts. Accepts files in any format readable by opencv. Can accept a list of 
images as inputs, and will process all of them sequentially. The list should be a text file with the
name of each image to be processed on a separate line.

Usage: edgedetect input_file [output_file] [options]

Options:

-h                 displays this information
-p                 use Prewitt filter
-r                 use Roberts filter
-sc                use Scharr filter (currently buggy)
-so                use Sobel filter (default)
-t                 display timing information to console
-x                 filter along x-axis (default)
-y                 filter along y-axis

Requirements:

OpenCV and CUDA

Archive contents:

bin               contains a compiled executable
data              contains examples of applying each filter to a sample image
src               contains the code
edgehelper.sh     a simple shell script for applying edgedetect to all image files in a folder
                  should be run in the same folder as edgedetect.exe
log.txt           output of running the program on the entire USC-SIPI database
makefile          used to compile the program
readme.txt        what you're reading now

Remarks: Right now there is a bug in the Shcarr filer that causes it to make the right half of the image blank.
I suspect that the npp library function itself is bugged and I would have to rewrite it from scratch to fix it.