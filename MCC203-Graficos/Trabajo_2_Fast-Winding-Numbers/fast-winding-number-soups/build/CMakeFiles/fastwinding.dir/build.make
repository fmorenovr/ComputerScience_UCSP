# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build

# Include any dependencies generated for this target.
include CMakeFiles/fastwinding.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fastwinding.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fastwinding.dir/flags.make

CMakeFiles/fastwinding.dir/main.cpp.o: CMakeFiles/fastwinding.dir/flags.make
CMakeFiles/fastwinding.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fastwinding.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fastwinding.dir/main.cpp.o -c /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/main.cpp

CMakeFiles/fastwinding.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastwinding.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/main.cpp > CMakeFiles/fastwinding.dir/main.cpp.i

CMakeFiles/fastwinding.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastwinding.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/main.cpp -o CMakeFiles/fastwinding.dir/main.cpp.s

CMakeFiles/fastwinding.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/fastwinding.dir/main.cpp.o.requires

CMakeFiles/fastwinding.dir/main.cpp.o.provides: CMakeFiles/fastwinding.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastwinding.dir/build.make CMakeFiles/fastwinding.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/fastwinding.dir/main.cpp.o.provides

CMakeFiles/fastwinding.dir/main.cpp.o.provides.build: CMakeFiles/fastwinding.dir/main.cpp.o


CMakeFiles/fastwinding.dir/utils/array.cpp.o: CMakeFiles/fastwinding.dir/flags.make
CMakeFiles/fastwinding.dir/utils/array.cpp.o: ../utils/array.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fastwinding.dir/utils/array.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fastwinding.dir/utils/array.cpp.o -c /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/array.cpp

CMakeFiles/fastwinding.dir/utils/array.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastwinding.dir/utils/array.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/array.cpp > CMakeFiles/fastwinding.dir/utils/array.cpp.i

CMakeFiles/fastwinding.dir/utils/array.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastwinding.dir/utils/array.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/array.cpp -o CMakeFiles/fastwinding.dir/utils/array.cpp.s

CMakeFiles/fastwinding.dir/utils/array.cpp.o.requires:

.PHONY : CMakeFiles/fastwinding.dir/utils/array.cpp.o.requires

CMakeFiles/fastwinding.dir/utils/array.cpp.o.provides: CMakeFiles/fastwinding.dir/utils/array.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastwinding.dir/build.make CMakeFiles/fastwinding.dir/utils/array.cpp.o.provides.build
.PHONY : CMakeFiles/fastwinding.dir/utils/array.cpp.o.provides

CMakeFiles/fastwinding.dir/utils/array.cpp.o.provides.build: CMakeFiles/fastwinding.dir/utils/array.cpp.o


CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o: CMakeFiles/fastwinding.dir/flags.make
CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o: ../utils/solidAngle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o -c /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/solidAngle.cpp

CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/solidAngle.cpp > CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.i

CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/utils/solidAngle.cpp -o CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.s

CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.requires:

.PHONY : CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.requires

CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.provides: CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastwinding.dir/build.make CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.provides.build
.PHONY : CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.provides

CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.provides.build: CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o


# Object files for target fastwinding
fastwinding_OBJECTS = \
"CMakeFiles/fastwinding.dir/main.cpp.o" \
"CMakeFiles/fastwinding.dir/utils/array.cpp.o" \
"CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o"

# External object files for target fastwinding
fastwinding_EXTERNAL_OBJECTS =

fastwinding: CMakeFiles/fastwinding.dir/main.cpp.o
fastwinding: CMakeFiles/fastwinding.dir/utils/array.cpp.o
fastwinding: CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o
fastwinding: CMakeFiles/fastwinding.dir/build.make
fastwinding: tbb/libtbb_static.a
fastwinding: /usr/lib/x86_64-linux-gnu/libGL.so
fastwinding: glad/libglad.a
fastwinding: glfw/src/libglfw3.a
fastwinding: /usr/lib/x86_64-linux-gnu/librt.so
fastwinding: /usr/lib/x86_64-linux-gnu/libm.so
fastwinding: /usr/lib/x86_64-linux-gnu/libX11.so
fastwinding: CMakeFiles/fastwinding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable fastwinding"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fastwinding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fastwinding.dir/build: fastwinding

.PHONY : CMakeFiles/fastwinding.dir/build

CMakeFiles/fastwinding.dir/requires: CMakeFiles/fastwinding.dir/main.cpp.o.requires
CMakeFiles/fastwinding.dir/requires: CMakeFiles/fastwinding.dir/utils/array.cpp.o.requires
CMakeFiles/fastwinding.dir/requires: CMakeFiles/fastwinding.dir/utils/solidAngle.cpp.o.requires

.PHONY : CMakeFiles/fastwinding.dir/requires

CMakeFiles/fastwinding.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fastwinding.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fastwinding.dir/clean

CMakeFiles/fastwinding.dir/depend:
	cd /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build /home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/CMakeFiles/fastwinding.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fastwinding.dir/depend

