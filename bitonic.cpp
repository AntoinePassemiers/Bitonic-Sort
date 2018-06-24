/**
    Sorting bitonic sequences using a distributed implementation 
    of the bitonic sort algorithm

    @author Antoine Passemiers
    @version 2.1 20/12/17
*/

#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "mpi.h"


/**
    Compare-swap operation on a sub-sequence.
    Each element i is compared with the element i+half.
    If the former is strictly less than the latter and the sorting
    order is descending, than the values are swapped.
    If the former is greater than the latter and the sorting
    order is ascending, than the values are swapped.

    @param sequence  Sequence or sub-sequence
    @param n_elements  Number of elements in the sequence
    @param ascending  Whether to sort in increasing order or not 
*/
void compareSwap(int* subsequence, int n_elements, bool ascending) {
    int temp, half = n_elements / 2; 
    for (int i = 0; i < half; i++) {
        if (ascending ^ (subsequence[i] < subsequence[i+half])) {
            // Basic swap operation
            temp = subsequence[i];
            subsequence[i] = subsequence[i+half];
            subsequence[i+half] = temp;
        }
    }
}

/**
    Creates a subset of node identifiers, and returns a lambda function
    that tells whether a node belongs to the subset. This is used to
    know whether a node is a receiver, a sender, or a currently
    inactive node.

    @param half  Half the size of the sequence to sort
    @param step  Dividor such that (half / step) is the number of nodes
                 between two adjacent nodes of the same subset
    @param offset  Node with the lowest identifier of the subset
    @return  Lambda function that returns true if a node is in the subset
*/
std::function<bool (int)> isInSubset(int half, int step, int offset) {
    std::vector<int> nodes;
    for (int i = 0; i < half; i += (half / step)) {
        nodes.push_back(i + offset);
    }
    return [nodes](int rank) { return (std::find(nodes.begin(), nodes.end(), rank) != nodes.end()); };
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, nb_instances;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_instances);
    MPI_Status status;

    int cnodes = nb_instances - 1;
    int n = cnodes * 2; // Size of the bitonic sequence to sort
    int m = n / 2; // Number of nodes
    int buf[n];
    bool ascending = true;
    int tag = 123; // Arbitrary tag

    if (n == 16) {
        if (rank == 0) {
            int A[n] = {14, 16, 15, 11, 9, 8, 7, 5, 4, 2, 1, 3, 6, 10, 12, 13};
            std::copy_n(A, n, buf); // Store sequence in buffer
        }
    } else {
        if (rank == 0) {
            // Generates a random bitonic sequence of the right size and shuffles it
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::iota(buf, buf + n, 0); // Fill buffer with an arange(0, n)
            std::shuffle(buf, buf + n, std::default_random_engine(seed));
            // Randomly split the sequence into two parts of unequal lengths
            int split = std::rand() % n;
            // Sort first part of the sequence in decreasing order
            std::sort(buf, buf + split, [](const int& lhs, const int& rhs){return lhs > rhs;});
            // Sort second part of the sequence in ascending order
            std::sort(buf + split, buf + n, [](const int& lhs, const int& rhs){return lhs < rhs;});
        }
    }

    if (rank == 0) {
        // First compare-swap iteration on n elements (can't be parallelized)
        compareSwap(buf, n, ascending);
    }

    int step = 1;
    while (m > 1) {
        // Lambda functions that tell whether rank is a sender/receiver or not
        std::function<bool (int)> isASender = isInSubset(n / 2, step, 0);
        std::function<bool (int)> isAReceiver = isInSubset(n / 2, step, m / 2);

        if (isASender(rank)) {
            int receiver = rank + (m / 2); // isAReceiver(rank+m/2) is then equal to true
            MPI_Send(&buf[m], m, MPI_INT, receiver, tag, MPI_COMM_WORLD);
            compareSwap(buf, m, ascending);
        } else if (isAReceiver(rank)) { 
            int sender = rank - (m / 2); // isASender(rank-m/2) is then equal to true
            MPI_Recv(buf, m, MPI_INT, sender, tag, MPI_COMM_WORLD, &status);
            compareSwap(buf, m, ascending);
        }

        m /= 2;
        step *= 2;
    }

    // Gathers the results from all slaves into the master node.
    // Each slave node contains two elements of the sequence.
    MPI_Gather(buf, 2, MPI_INT, buf, 2, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize(); // MPI is no longer required from here

    if (rank == 0) {
        // Display the sorted sequence
        std::cout << "Sorted sequence : ";
        for (int i = 0; i < n; i++)
            std::cout << buf[i] << " ";
        std::cout << std::endl;
    }
    return 0;
}

