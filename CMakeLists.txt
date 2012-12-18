CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
FILE(TO_CMAKE_PATH "$ENV{IMAGINEPP_ROOT}" d)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${d}/CMake")
FIND_PACKAGE(Imagine)

PROJECT(Deblur)

ImagineAddExecutable(Deblur
        Deblur.cpp
		)
ImagineUseModules(Deblur LinAlg Graphics Images)
