# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(eigen-download NONE)

include(ExternalProject)
ExternalProject_Add(eigen-download
                    GIT_CONFIG advice.detachedHead=false;URL;http://bitbucket.org/eigen/eigen/get/3.2.10.tar.gz;URL_MD5;8ad10ac703a78143a4062c9bda9d8fd3
                    SOURCE_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/libigl/cmake/../external/eigen"
                    BINARY_DIR          "/home/jenazads/github/ComputerScience_UCSP/MCC203_Graficos/Trabajo_2_Fast-Winding-Numbers/fast-winding-number-soups/build/eigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
