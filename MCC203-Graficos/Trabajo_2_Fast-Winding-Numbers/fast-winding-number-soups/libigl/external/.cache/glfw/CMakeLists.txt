# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(glfw-download NONE)

include(ExternalProject)
ExternalProject_Add(glfw-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/glfw/glfw.git;GIT_TAG;58cc4b2c5c2c9a245e09451437dd6f5af4d60c84
                    SOURCE_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/libigl/cmake/../external/glfw"
                    BINARY_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/glfw-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
