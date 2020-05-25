cmake_minimum_required(VERSION 2.8)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

project(opencv_light_corrector)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} "main.cpp" "lightcorrector.cpp" "lightcorrector.h")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
