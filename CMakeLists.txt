cmake_minimum_required(VERSION 3.16)

# 显卡架构：10系=61，20系=75，30系=86，40系=89
set(CMAKE_CUDA_ARCHITECTURES 61)

# ========== CUDA宿主编译器配置（仅Linux/WSL环境保留，Windows MSVC请删除本段） ==========
# CMake官方推荐方式：指定CUDA宿主编译器为gcc-12
set(CMAKE_CUDA_HOST_COMPILER gcc-12)
# 兜底参数：直接向nvcc传递-ccbin，双重保证编译器生效
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin gcc-12")

# 项目声明：必须在根project中启用CUDA语言，这是所有CUDA编译的合法入口
project(my_torch LANGUAGES CXX CUDA)

# C++ 标准配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 标准配置，与C++标准对齐
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========== 核心修复：按构建类型区分编译选项 ==========
# 已删除原全局硬编码的 -O2 -DNDEBUG，避免Debug模式被强制优化
# --- Debug 调试模式：关闭所有优化，生成完整调试符号 ---
# 主机端C++：CMake默认自动开启 -O0 + 调试符号，无需额外设置
# CUDA端：补充设备端源码调试符号(-G) + 主机端完整符号(-g) + 全量关闭优化(-O0)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")

# --- Release 发布模式：保持原O2优化级别，保证运行性能 ---
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2 -DNDEBUG")

# 源文件：自动匹配所有.cpp与.cu文件，后续新增算子文件自动纳入编译
file(GLOB SRC_FILES
    src/*.cpp
    src/*.cu
)

# 生成可执行文件
add_executable(my_torch ${SRC_FILES})

# 头文件路径
target_include_directories(my_torch PRIVATE include)

# 链接CUDA运行时库，所有CUDA API调用都依赖它
target_link_libraries(my_torch PRIVATE cudart)

# 兼容当前写法：指定tensor.cpp用CUDA编译器编译
# 后续若将GPU算子拆分到独立的.cu文件中，可直接删除此行
set_source_files_properties(src/tensor.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/operators.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/utils.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/operators.cu PROPERTIES LANGUAGE CUDA)
