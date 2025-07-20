cmake_minimum_required(VERSION 3.23)
project(pathCXX2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(pathCXX2 main.cpp path.cpp)

if (0)
find_package(SFML COMPONENTS Network Graphics Window Audio System CONFIG REQUIRED)
target_link_libraries(pathCXX2 PRIVATE SFML::Network SFML::Graphics SFML::Window SFML::Audio SFML::System)

find_package(Poco COMPONENTS Foundation REQUIRED)
target_link_libraries(pathCXX2 PRIVATE Poco::Foundation)
endif ()
