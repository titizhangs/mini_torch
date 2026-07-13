cmake_minimum_required(VERSION 3.16)
project(my_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 源文件
file(GLOB SRC_FILES
    src/*.cpp
)


# 可执行文件
add_executable(my_torch
    src/main.cpp
    ${SRC_FILES}
)

#包含头文件
target_include_directories(my_torch PRIVATE include)

set(CMAKE_BUILD_TYPE Debug)          # 生成调试符号
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # -g 加调试信息，-O0 关闭优化