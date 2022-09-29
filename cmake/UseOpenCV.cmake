function(useOpenCV TARGET_NAME)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${TARGET_NAME} PUBLIC "-framework Carbon")
    target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

    target_link_libraries(${TARGET_NAME} PUBLIC
            ${tesseract_lib}
            ${leptonica_lib}
            ${TESSERACT_LIBRARIES}
            ${LEPTONICA_LIBRARIES}
            )

    target_include_directories(${TARGET_NAME} PUBLIC
            ${tesseract_include}
            ${leptonica_include})
endfunction()
