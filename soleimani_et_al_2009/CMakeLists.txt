CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(myproject)



ADD_LIBRARY(mylib /home/c1045890/libraries/SurfaceCoefficients.cpp /home/c1045890/libraries/AnalyticSolution.cpp /home/c1045890/libraries/MaterialData.cpp /home/c1045890/libraries/BoundaryConditions.cpp)
DEAL_II_SETUP_TARGET(mylib)


ADD_EXECUTABLE(mycode soleimani_et_al_2009.cc)
DEAL_II_SETUP_TARGET(mycode)

TARGET_LINK_LIBRARIES(mycode mylib)


ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )
