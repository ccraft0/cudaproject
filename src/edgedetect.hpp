#include <iostream>
#include <fstream>
#include <tuple>
#include <string>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <npp.h>
#include <nppi_support_functions.h>

std::tuple<std::string, std::string, int, bool, bool> parseCommandLineArguments(int argc, char** argv);
std::string makeOutputFileName(std::string input_filename);
int readInputFileList(std::string input_filename, std::string output_filename, int filter_type, bool y_axis, bool show_timing);
int processImage(std::string input_filename, std::string output_filename, int filter_type, bool y_axis, bool show_timing);