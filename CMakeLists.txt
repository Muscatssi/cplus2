cmake_minimum_required(VERSION 3.10)
project(plate_ocr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# Tesseract
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)
include_directories(${TESSERACT_INCLUDE_DIRS})

# 실행파일 설정
add_executable(plate_ocr main.cpp)

# 라이브러리 링크
target_include_directories(plate_ocr PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    $ENV{HOME}/vcpkg/installed/arm64-linux/include
)
target_link_libraries(plate_ocr PRIVATE 
    ${OpenCV_LIBS} 
    ${TESSERACT_LIBRARIES}
)