cmake_minimum_required(VERSION 2.8)
Project(kinect2_pclgrabber CXX)

set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE  Release)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/modules")
set(INCLUDE           "${CMAKE_SOURCE_DIR}/utils")

option(WITH_OPENGL "adds opengl support for libfreenect2" OFF)
IF(${WITH_OPENGL})
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
ENDIF()

option(WITH_OPENCV "adds opencv support for libfreenect2" ON)
IF(${WITH_OPENCV})
  find_package(OpenCV REQUIRED)
ENDIF()

option(WITH_OPENCL "adds opencl support for freenect2" OFF)
if(${WITH_OPENCL})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_OPENCL")
endif()

option(WITH_CUDA "adds cuda support for freenect2" OFF)
if(${WITH_CUDA})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_CUDA")
endif()

option(WITH_SERIALIZATION "adds support for boost serialization" OFF)
if(${WITH_SERIALIZATION})
	find_package(Boost REQUIRED)
	include_directories(${INCLUDE})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_SERIALIZATION")
	set(BOOST_LIBS "boost_serialization")
endif()

option(WITH_ROS "adds support for ROS" OFF)
if(${WITH_ROS})
	find_package(catkin REQUIRED COMPONENTS   roscpp std_msgs pcl_conversions  pcl_ros
						sensor_msgs cv_bridge image_transport tf
						message_generation )
	catkin_package( CATKIN_DEPENDS roscpp rospy std_msgs pcl_conversions  pcl_ros
						sensor_msgs cv_bridge image_transport tf
						cmake_modules
	)
	include_directories(${catkin_INCLUDE_DIRS} include)
endif()

option(WITH_PCL "adds pcl cloud support" ON)
if(${WITH_PCL})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_PCL")
	find_package(PCL REQUIRED)
	link_directories(${PCL_LIBRARY_DIRS})
	add_definitions(${PCL_DEFINITIONS})
	include_directories(${PCL_INCLUDE_DIRS})
endif()

find_package(Freenect2  REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FREENECT2_INCLUDE_DIRS})
include_directories(${INCLUDE})

add_executable(Kinect2_PCLGrabber      fast3DRecognition.cpp)
#add_executable(Kinect2_Grabber         test_kinect2.cpp)
#add_executable(MultiKinect2_PCLGrabber test_multi_kinect2.cpp)

#target_link_libraries(Kinect2_Grabber         ${OpenCV_LIBS} ${FREENECT2_LIBRARY} $)
target_link_libraries(Kinect2_PCLGrabber      ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${PCL_LIBRARIES} ${BOOST_LIBS})
#target_link_libraries(MultiKinect2_PCLGrabber ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${PCL_LIBRARIES} ${BOOST_LIBS})

if(${WITH_ROS})
	add_executable(RosKinect2_PCLGrabber test_ros_kinect2.cpp)
	target_link_libraries(RosKinect2_PCLGrabber ${catkin_LIBRARIES} ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${PCL_LIBRARIES} ${BOOST_LIBS})
endif()

