#pragma once

#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <array>

template<typename T>
std::vector<T> random_matrix(int i, int j, unsigned int seed); 

template<typename T>
std::vector<int> random_array(T min, T max, unsigned int N, unsigned int seed = 1999);

#ifdef PPROG_IMPLEMENTATION

template<>
std::vector<int> random_matrix<int>(int i, int j, unsigned int seed) {

    std::minstd_rand engine(seed);
    std::uniform_int_distribution<int> distr(1, 10);

    std::vector<int> result;
    result.resize(i*j);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

template<>
std::vector<float> random_matrix<float>(int i, int j, unsigned int seed) {
    std::minstd_rand engine(seed);
    std::uniform_real_distribution<float> distr(0.0, 10.0);

    std::vector<float> result;
    result.resize(i*j);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

template<>
std::vector<int> random_array(int min, int max, unsigned int N, unsigned int seed) {
    std::minstd_rand engine(seed);
    std::uniform_int_distribution<int> distr(min, max);

    std::vector<int> result(N);
    std::generate(result.begin(), result.end(), [&]() {
        return distr(engine);
    });

    return result;
}

#endif