# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(glad-download NONE)

include(ExternalProject)
ExternalProject_Add(glad-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/libigl-glad.git;GIT_TAG;71e35fe685a0cc160068a2f2f971c40b82d14af0
                    SOURCE_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/libigl/cmake/../external/glad"
                    BINARY_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/glad-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
