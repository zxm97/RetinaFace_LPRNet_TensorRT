cmake_minimum_required(VERSION 2.6)

project(retinaface) # 1
set(OpenCV_DIR "D:\\my_opencv\\build")  #2
set(TRT_DIR "D:\\TensorRT-8.5.1.7")  #3
message(PROJECT SOURCE DIRECTORY "    libraries: ${PROJECT_SOURCE_DIR}")
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}\\include)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast  -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
# set_target_properties(decodeplugin PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(decodeplugin nvinfer cudart)
link_directories(${TRT_DIR}\\lib)
# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${TRT_DIR}\\lib)


#add_executable(retina_r50 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_r50.cpp)

#target_link_libraries(retina_r50 nvinfer)
#target_link_libraries(retina_r50 cudart)
#target_link_libraries(retina_r50 decodeplugin)
#target_link_libraries(retina_r50 ${OpenCV_LIBRARIES})

#add_executable(retina_mnet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_mnet.cpp)

#target_link_libraries(retina_mnet nvinfer)
#target_link_libraries(retina_mnet cudart)
#target_link_libraries(retina_mnet decodeplugin)
#target_link_libraries(retina_mnet ${OpenCV_LIBRARIES})

add_executable(retina_mnet_plate ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_mnet_plate.cpp) #5

target_link_libraries(retina_mnet_plate nvinfer)
target_link_libraries(retina_mnet_plate cudart)
target_link_libraries(retina_mnet_plate decodeplugin)
target_link_libraries(retina_mnet_plate ${OpenCV_LIBRARIES})


add_executable(LPRnet_multiline ${PROJECT_SOURCE_DIR}/LPRnet_multiline.cpp)
target_link_libraries(LPRnet_multiline nvinfer)
target_link_libraries(LPRnet_multiline cudart)
#target_link_libraries(LPRnet_multiline ${OpenCV_LIBS})
target_link_libraries(LPRnet_multiline ${OpenCV_LIBRARIES})


add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(demo nvinfer)
target_link_libraries(demo cudart)
target_link_libraries(demo decodeplugin)
target_link_libraries(demo ${OpenCV_LIBRARIES})

add_executable(demo_mt ${PROJECT_SOURCE_DIR}/demo_mt.cpp)
target_link_libraries(demo_mt nvinfer)
target_link_libraries(demo_mt cudart)
target_link_libraries(demo_mt decodeplugin)
target_link_libraries(demo_mt ${OpenCV_LIBRARIES})

# add_definitions(-O2 -pthread)

