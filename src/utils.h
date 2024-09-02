#include <iostream>
#include <concepts>

template <typename T>
concept Number = std::floating_point<T>;