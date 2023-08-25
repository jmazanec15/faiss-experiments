//
// Created by Mazanec, Jack on 8/25/23.
//

#include <iostream>
#include "faiss/IndexFlat.h"


int main() {
    std::cout << "Hello World" << std::endl;
    auto * index = new faiss::IndexFlat();

    return 0;
}
