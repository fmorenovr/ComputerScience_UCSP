find_path(FREENECT2_INCLUDE_DIRS NAMES libfreenect2.hpp
	HINTS
	/usr/local/include/libfreenect2/
	/usr/include/libfreenect2
	/usr/local/include/
	/usr/include/
	}
)
 
find_library(FREENECT2_LIBRARY NAMES freenect2)

if(FREENECT2_INCLUDE_DIRS AND FREENECT2_LIBRARY)
  set(FREENECT2_FOUND TRUE)
endif()

if(FREENECT2_LIBRARY)
    set(FREENECT2_LIBRARY ${FREENECT2_LIBRARY})
endif()

if (FREENECT2_FOUND)
  mark_as_advanced(FREENECT2_INCLUDE_DIRS FREENECT2_LIBRARY FREENECT2_LIBRARIES)
endif()
