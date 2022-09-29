find_path(ONNXRUNTIME_INCLUDE_DIR
	NAMES onnxruntime_c_api.h
	PATHS
		${ONNXRUNTIME_ROOT}/include
		/usr
		/usr/local
	PATH_SUFFIXES include
	NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_LIBRARY
	NAMES onnxruntime
	PATHS
		${ONNXRUNTIME_ROOT}/lib
		/usr
		/usr/local
	PATH_SUFFIXES lib
	NO_DEFAULT_PATH
)

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ONNXRUNTIME
	REQUIRED_VARS
		ONNXRUNTIME_LIBRARY
		ONNXRUNTIME_INCLUDE_DIR
)
