cmake_minimum_required(VERSION 3.0)
project(ADPCCSO_optimization)

# Устанавливаем стандарт C++ (например, C++20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release) # Тип сборки в Debug
add_definitions(-O3) # Флаги

# Добавляем исходные файлы
add_executable(
    ${PROJECT_NAME}
    solution/agent.cpp
    solution/agent.h
    solution/agents_initializer.cpp
    solution/agents_initializer.h
    solution/agent_t.cpp
    solution/agent_t.h
    solution/chick.cpp
    solution/chick.h
    solution/chicken.cpp
    solution/chicken.h
    solution/chicken_swarm.cpp
    solution/chicken_swarm.h
    solution/fish_swarm.cpp
    solution/fish_swarm.h
    solution/fish.cpp
    solution/fish.h
    solution/fitness_function.cpp
    solution/fitness_function.h
    solution/fitness_functions/alpine_1.cpp
    solution/fitness_functions/alpine_1.h
    solution/fitness_functions/alpine_2.cpp
    solution/fitness_functions/alpine_2.h
    solution/fitness_functions/deflected_corrugated_spring.cpp
    solution/fitness_functions/deflected_corrugated_spring.h
    solution/fitness_functions/high_load.cpp
    solution/fitness_functions/high_load.h
    solution/fitness_functions/rastrigin.cpp
    solution/fitness_functions/rastrigin.h
    solution/fitness_functions/spherical.cpp
    solution/fitness_functions/spherical.h
    solution/fitness_functions/main_task_cube.cpp
    solution/fitness_functions/main_task_cube.h
    solution/fitness_functions/main_task_simplex.cpp
    solution/fitness_functions/main_task_simplex.h
    solution/fitness_functions/main_task_sphere.cpp
    solution/fitness_functions/main_task_sphere.h
    solution/helpers.cpp
    solution/helpers.h
    solution/hen.cpp
    solution/hen.h
    solution/hypercube.cpp
    solution/hypercube.h
    solution/main.cpp
    solution/rootster.cpp
    solution/rootster.h
    solution/swarm.cpp
    solution/swarm.h
)

target_include_directories(${PROJECT_NAME} PRIVATE "./libs")
target_include_directories(${PROJECT_NAME} PRIVATE "./solution")

# Ищем библиотеки
find_package(Curses REQUIRED)
find_package(GSL REQUIRED)

# Добавление опций сборки
target_compile_options(${PROJECT_NAME} PRIVATE)

# Включение директорий заголовочных файлов
target_include_directories(${PROJECT_NAME} PRIVATE ${CURSES_INCLUDE_DIR})

# Связываем проект с библиотеками
target_link_libraries(${PROJECT_NAME} GSL::gsl GSL::gslcblas ${CURSES_LIBRARIES})