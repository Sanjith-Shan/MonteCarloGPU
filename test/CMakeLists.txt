include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(mc_tests
    test_european.cpp
    test_asian.cpp
    test_greeks.cpp
    test_convergence.cpp
)
target_link_libraries(mc_tests PRIVATE mc_engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mc_tests)
