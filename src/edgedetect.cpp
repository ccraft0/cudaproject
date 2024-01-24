#include "edgedetect.hpp"
#define SCHARR_FILTER 0
#define SOBEL_FILTER 1
#define PREWITT_FILTER 2
#define ROBERTS_FILTER 3

int main (int argc, char** argv) {
    // Retreive command line arguments
    auto clargs = parseCommandLineArguments(argc, argv);

    std::string input_filename = std::get<0>(clargs);
    std::string output_filename = std::get<1>(clargs);
    int filter_type = std::get<2>(clargs);
    bool y_axis = std::get<3>(clargs);
    bool show_timing = std::get<4>(clargs);

    // If command line arguments are invalid, or user uses -h, output usage information
    if (input_filename.length() == 0) {
        std::cout << R"USAGE(Description: Takes an input file and implements edge detection using one of four filters, either
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
)USAGE";
        return EXIT_SUCCESS;
    }

    // Output name of device
    // REMARK: This was originally intended as a debug statement, to find out if the video card was working at all.
    // It is being kept because for some reason, the program only works when this statement is included.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.name << std::endl;
    
    // If input filename ends in .txt, treat input as list of files
    if (input_filename.length() >= 4 && input_filename.substr(input_filename.length() - 4, 4) == ".txt") {
        readInputFileList(input_filename, output_filename, filter_type, y_axis, show_timing);
    }
    // Otherwise, process as a single image
    else {
        processImage(input_filename, output_filename, filter_type, y_axis, show_timing);
    }
    return EXIT_SUCCESS;
}

int processImage(std::string input_filename, std::string output_filename, int filter_type, bool y_axis, bool show_timing) {
    // Initialize timer
    std::chrono::high_resolution_clock timer;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = timer.now();

    // If output_filename is not specified, choose one automatically
    if (output_filename.length() == 0) {
        output_filename = makeOutputFileName(input_filename);
    }
    
    // Open input file
    cv::Mat input_image = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

    // Output error message if image failed to load
    if (input_image.data == NULL) {
        std::cout << "Input image " + input_filename + " failed to load.\n";
        return EXIT_FAILURE;
    }
    std::cout << "Input image " + input_filename + " loaded.\n";
    
    // Record time taken to load image
    std::chrono::time_point<std::chrono::high_resolution_clock> load_time = timer.now();
    
    // Determine size of image
    int input_rows = input_image.rows;
    int input_columns = input_image.cols;
    int input_length = input_image.total();
    int input_element_size = input_image.elemSize();
    int host_input_stride = input_element_size*input_columns;

    // Extract data from image. If the matrix is continuous, this can be done without any data copies.
    // I don't know whether the matrix returned by imread is ever *not* continuous, but it seems wise not to assume it.
    uchar* input_pointer;
    if (input_image.isContinuous()) {
        input_pointer = input_image.data;
        // DEBUG
        // std::cout << "Input image is continuous" << std::endl;
    }
    else {
        input_pointer = (uchar*) malloc(input_length*input_element_size);
        for (int i = 0; i < input_rows; i++) {
            memcpy(input_pointer + i*host_input_stride, input_image.ptr(i), input_element_size);
        }
      // DEBUG
      // std::cout << "Input image is not continuous" << std::endl;
    }

    // DEBUG: Output filter type
    // switch (filter_type) {
    //    case SCHARR_FILTER:
    //        std::cout << "using Scharr filter\n";
    //        break;
    //    case SOBEL_FILTER:
    //        std::cout << "using Sobel filter\n";
    //        break;
    //    case PREWITT_FILTER:
    //        std::cout << "using Prewitt filter\n";
    //        break;
    //    case ROBERTS_FILTER:
    //        std::cout << "using Roberts filter\n";
    // }

    // Copy data to device
    Npp8u* dev_input_pointer;
    int dev_input_stride;
    // DEBUG
    // std::cout << "Device input stride before allocation: " << dev_input_stride << std::endl;
    dev_input_pointer = nppiMalloc_8u_C1(input_columns, input_rows, &dev_input_stride);
    
    cudaError_t err;
    err = cudaMemcpy2D(dev_input_pointer, dev_input_stride, input_pointer, host_input_stride, host_input_stride, input_rows, cudaMemcpyHostToDevice);
    
    // Record memory transfer time
    std::chrono::time_point<std::chrono::high_resolution_clock> memcpyHostToDevice_time = timer.now();

    // DEBUG statements
    // std::cout << "Dev input pointer: " << (void*) dev_input_pointer << std::endl;
    // std::cout << "Dev input stride: " << dev_input_stride << std::endl;
    // std::cout << "Input pointer: " << (void*) input_pointer << std::endl;
    // std::cout << "Host input stride: " << host_input_stride << std::endl;
    // std::cout << "Input rows: "  << input_rows << std::endl;

    if (err != cudaSuccess) {
        std::cout << "Failed to copy input image to device.\n";
        std::cout << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // DEBUG: copy memory back out and output immediately
    // uchar* DEBUG_output_pointer;
    // int DEBUG_output_stride = host_input_stride;
    // int DEBUG_output_size = input_rows*DEBUG_output_stride;
    // DEBUG_output_pointer = (uchar*) malloc(DEBUG_output_size);
    // cudaMemcpy2D(DEBUG_output_pointer, DEBUG_output_stride, dev_input_pointer, dev_input_stride, host_input_stride, input_rows, cudaMemcpyDeviceToHost);
    // cv::Mat DEBUG_output_image = cv::Mat(input_rows, input_columns, CV_8UC1, (void*) DEBUG_output_pointer, DEBUG_output_stride);
    // cv::imwrite("DEBUG.jpg", DEBUG_output_image);
    // free(DEBUG_output_pointer);

    // Define region of interest. Of course this is the entire image.
    NppiSize ROI = {input_columns, input_rows};
    
    // Run appropriate kernel. Scharr filter is handled in a different block than the other filters, because all versions of
    // the Scharr filter output at least 16 bits per pixel.
    Npp8u* dev_output_pointer;
    Npp16s* dev_output_pointer_16;
    int dev_output_stride;
    if (filter_type == SCHARR_FILTER) {
        // Allocate output memory
        dev_output_pointer_16 = nppiMalloc_16s_C1(input_columns, input_rows, &dev_output_stride);

        // Apply kernel
        if (y_axis) {
            // DEBUG
            NppStatus status;
            status = nppiFilterScharrVertBorder_8u16s_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer_16, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
            std::cout << "Npp status code is: " << status << std::endl;
        }
        else {
            // DEBUG
            NppStatus status;
            status = nppiFilterScharrHorizBorder_8u16s_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer_16, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
            std::cout << "Npp status code is: " << status << std::endl;
        }
    }
    else {
        // Allocate output memory
        dev_output_pointer = nppiMalloc_8u_C1(input_columns, input_rows, &dev_output_stride);

        // Apply kernel
        switch (filter_type) {
            case SOBEL_FILTER:        
                if (y_axis) {
                    nppiFilterSobelVertBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                else {
                    nppiFilterSobelHorizBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                break;
            case PREWITT_FILTER:
                if (y_axis) {
                    nppiFilterPrewittVertBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                else {
                    nppiFilterPrewittHorizBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                break;
            case ROBERTS_FILTER:
                if (y_axis) {
                    nppiFilterRobertsUpBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                else {
                    nppiFilterRobertsDownBorder_8u_C1R(dev_input_pointer, dev_input_stride, ROI, {0, 0}, dev_output_pointer, dev_output_stride, ROI, NPP_BORDER_REPLICATE);
                }
                break;
                std::cout << "This line should be unreachable. If you are reading this, something has gone terribly wrong.";
        }
    }

    // Record kernel execution time
    std::chrono::time_point<std::chrono::high_resolution_clock> kernel_time = timer.now();

    // Copy device memory to host memory
    uchar* output_pointer;
    int output_stride;

    if (filter_type == SCHARR_FILTER) {
        output_stride = input_columns*sizeof(Npp16s);
    }
    else {
        output_stride = input_columns;
    }

    // DEBUG
    // std::cout << "Output stride is: " << output_stride << std::endl;
    // std::cout << "Device output stride is: " << dev_output_stride << std::endl;

    int output_size = input_rows*output_stride;
    output_pointer = (uchar*) malloc(output_size);

    // Need to split into two cases, since dev_output_pointer and dev_output_pointer_16 have different types.
    if (filter_type == SCHARR_FILTER) {
        err = cudaMemcpy2D(output_pointer, output_stride, dev_output_pointer_16, dev_output_stride, input_columns, input_rows, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "Failed to copy output image from device to host.\n";
        }
    }
    else {
        err = cudaMemcpy2D(output_pointer, output_stride, dev_output_pointer, dev_output_stride, input_columns, input_rows, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cout << "Failed to copy output image from device to host.\n";
        }
    }

    // Device memory is no longer needed
    nppiFree(dev_input_pointer);
    if (filter_type == SCHARR_FILTER) {
        nppiFree(dev_output_pointer_16);
    }
    else {
        nppiFree(dev_output_pointer);
    }

    // Record memory transfer time
    std::chrono::time_point<std::chrono::high_resolution_clock> memcpyDeviceToHost_time = timer.now();

    // Create output matrix
    cv::Mat output_image;
    if (filter_type == SCHARR_FILTER) {
        output_image = cv::Mat(input_rows, input_columns, CV_16SC1, (void*) output_pointer, output_stride);
    }
    else {
        output_image = cv::Mat(input_rows, input_columns, CV_8UC1, (void*) output_pointer, output_stride);
    }

    // Write output image to disk
    bool write_success = cv::imwrite(output_filename, output_image);
    if (write_success) {
        std::cout << "Output image " + output_filename + " successfully written.\n";
    }
    else {
        std::cout << "Output image " + output_filename + " could not be written.\n";
        exit(EXIT_FAILURE);
    }

    // Free host memory
    // free (input_pointer); -- redundant, since this is automatically done when the destructor for input_image is called
    free(output_pointer);

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = timer.now();

    if (show_timing) {
        std::cout << "Image processing took " + std::to_string((end_time - start_time).count()) + " nanoseconds:\n";
        std::cout << "  " + std::to_string((load_time - start_time).count()) + " to load the image.\n";
        std::cout << "  " + std::to_string((memcpyHostToDevice_time - load_time).count()) + " to transfer it to the device.\n";
        std::cout << "  " + std::to_string((kernel_time - memcpyHostToDevice_time).count()) + " to execute the kernel.\n";
        std::cout << "  " + std::to_string((memcpyDeviceToHost_time - kernel_time).count()) + " to transfer it back to the host.\n";
        std::cout << "  " + std::to_string((end_time - memcpyHostToDevice_time).count()) + " to write the image to disk.\n";
    }
    
    return EXIT_SUCCESS;
}

std::tuple<std::string, std::string, int, bool, bool> parseCommandLineArguments(int argc, char** argv) {
    std::string input_file = "";
    std::string output_file = "";
    int filter_type = SOBEL_FILTER;
    bool y_axis = false;
    bool show_timing = false;
    
    // If first argument is an option or doesn't exist, return empty input filename
    if (argc <= 1 || argv[1][0] == '-') {
        return {"", "", 0, false, false};
    }
    // otherwise, assume that the first argument is an input filename
    else {
        input_file = argv[1];
    }
    
    // Set argument pointer
    int argp = 2;

    // If the second argument exists and is not an option, it is the output filename
    if (argc > 2 && argv[2][0] != '-') {
        output_file = argv[2];
        argp++;
    }

    // Iterate over the remaining arguments, and set options accordingly
    while (argp < argc) {
        if (!strcmp(argv[argp], "-h")) {
            return {"", "", 0, false, false};
        }
        else if (!strcmp(argv[argp], "-p")) {
            filter_type = PREWITT_FILTER;
        }
        else if (!strcmp(argv[argp], "-r")) {
            filter_type = ROBERTS_FILTER;
        }
        else if (!strcmp(argv[argp], "-sc")) {
            filter_type = SCHARR_FILTER;
        }
        else if (!strcmp(argv[argp], "-so")) {
            filter_type = SOBEL_FILTER;
        }
        else if (!strcmp(argv[argp], "-t")) {
            show_timing = true;
        }
        else if (!strcmp(argv[argp], "-x")) {
            y_axis = false;
        }
        else if (!strcmp(argv[argp], "-y")) {
            y_axis = true;
        }
        else {
            std::cout << "Invalid argument to edgedetect. For usage information, use \"edgedetect -h\"\n";
            exit(EXIT_SUCCESS);
        }
        argp++;
    }

    // Return parsed arguments
    return {input_file, output_file, filter_type, y_axis, show_timing};

}

// Automatically generate output filename by adding "_edges" to the name of the input
std::string makeOutputFileName(std::string input_file) {
    int separator = input_file.find_last_of('.');
    if (separator == std::string::npos) {
        return input_file + "_edges";
    }
    else {
        std::string prefix = input_file.substr(0, separator);
        std::string extension = input_file.substr(separator+1, input_file.length());
        return prefix + "_edges." + extension;
    }
}

// Reads lists of input and output filenames to process
int readInputFileList (std::string input_file, std::string output_file, int filter_type, bool y_axis, bool show_timing) {
    std::ifstream inputs;
    std::ifstream outputs;
    bool output_file_exists = (output_file.length() != 0);

    // Open input file list
    inputs.open(input_file, std::ios_base::in);
    if (!inputs.good()) {
        std::cout << "Error opening input file list.\n";
        exit(EXIT_FAILURE);
    }

    // Open output_file list, if there is one
    if (output_file_exists) {
        outputs.open(output_file, std::ios_base::in);
        if (!outputs.good()) {
            std::cout << "Error opening output file list.\n";
            inputs.close();
            exit(EXIT_FAILURE);
        }
    }

    // While lines remain in the input file, process the images 
    std::string next_input = "";
    std::string next_output = "";
    while (getline(inputs, next_input)) {
        getline(outputs, next_output);
        processImage(next_input, next_output, filter_type, y_axis, show_timing);
    }
    return EXIT_SUCCESS;
}