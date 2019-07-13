file(GLOB NN_INCLUDE "include/*.hpp")
file(GLOB NN_SRC "src/*.cpp")

source_group("Header Files" FILES ${NN_INCLUDE})
source_group("Source Files" FILES ${NN_SRC})

add_library(NN
	${NN_INCLUDE}
	${NN_SRC}
)

target_include_directories(NN
	PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
	PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(NN PRIVATE Eigen3::Eigen)