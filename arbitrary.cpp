/**
    Sorting arbitrary sequences using a distributed implementation 
    of the bitonic sort algorithm

    @author Antoine Passemiers
    @version 2.1 20/12/17
*/

#include <algorithm>
#include <iostream>
#include <functional>
#include <numeric>
#include <vector>
#include <chrono>
#include <random>
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

    @param half  Half the size of the sub-sequence to sort
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
    return [nodes](int rank) { return (std::find(nodes.begin(), nodes.end(), rank) != nodes.end()); };;
}

/**
    Sorts a sub-sequence by assuming that it is bitonic. The sub-sequence is stored in the
    sub-master node, whose identifier is given as a parameter.

    @param buf  Buffer to receive and send part of the sub-sequence
    @param n  Size of the sub-sequence to sort
    @param master_node  Node identifier that plays the role of the master until the sub-sequence is sorted
    @param rank  Current node identifier
    @param ascending  Whether to sort the sub-sequence in ascending order or not
*/
void bitonicSort(int* buf, int n, int master_node, int rank, bool ascending, MPI_Status& status) {
    int tag = 123; // Arbitrary tag
    int m = n / 2; // Number of nodes involved in the sub-sequence sort
    if (rank == master_node) {
        // First compare-swap iteration on n elements (can't be parallelized)
        compareSwap(buf, n, ascending);
    }

    int step = 1;
    while (m > 1) {
        // Lambda functions that tell whether rank is a sender/receiver or not
        std::function<bool (int)> isASender = isInSubset(n / 2, step, master_node);
        std::function<bool (int)> isAReceiver = isInSubset(n / 2, step, master_node + (m / 2));

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
    
    // Manually gather the results from all slaves into the sub-master node
    // Each slave node contains two elements of the sub-sequence
    // For optimization purposes, The sub-master node does not send any
    // element to itself.
    if (rank != master_node) {
        // If the current node is a slave, send the two elements to the sub-master node
        MPI_Send(buf, 2, MPI_INT, master_node, tag, MPI_COMM_WORLD);
    } else {
        // If the current node is the sub-master, receive from each slave node except itself
        for (int i = 1; i < (n / 2); i++) {
            MPI_Recv(&buf[2 * i], 2, MPI_INT, master_node + i, tag, MPI_COMM_WORLD, &status);
        }
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, nb_instances;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_instances);
    MPI_Status status;
    int tag = 123; // Arbitrary tag

    int cnodes = nb_instances - 1;
    int n = cnodes * 2; // Number of nodes
    int buf[n]; // Buffer for sending and receiving sub-sequences

    // Initialisation of the arbitrary sequence to sort.
    // This is done in master node to avoid contamination.
    if (n == 16) {
        if (rank == 0) {
            int A[n] = {10, 6, 14, 11, 9, 16, 3, 13, 8, 12, 5, 2, 4, 15, 1, 7};
            std::copy_n(A, n, buf); // Store sequence in buffer
        }
    } else {
        if (rank == 0) {
            // Generates a random sequence of the right size and shuffles it
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::iota(buf, buf + n, 0);
            std::shuffle(buf, buf + n, std::default_random_engine(seed));
        }
    }

    // Scatters the sequence and applies a compare-swap operation on
    // pairs of element. One out of every two node applies the compare-swap
    // in ascending order and one out of every two applies it in descending order.
    // This is to create contiguous bitonic sequences of size 4.
    MPI_Scatter(buf, 2, MPI_INT, buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
    compareSwap(buf, 2, (rank % 2 == 0));


    int k = 4;
    while (k <= n) {

        // Merge
        for (int i = 0; i < (n / 2); i += (k / 4)) {
            if (rank == i) {
                int master_node = (i % (k / 2) == 0) ? i : i - (k / 4);
                MPI_Send(buf, (k / 2), MPI_INT, master_node, tag, MPI_COMM_WORLD);
                if (rank == master_node) {
                    // Receive the first half of the sub-sequence
                    MPI_Recv(buf, (k / 2), MPI_INT, i, tag, MPI_COMM_WORLD, &status);
                    // Receive the second half of the sub-sequence
                    MPI_Recv(&buf[k / 2], (k / 2), MPI_INT, i + (k / 4), tag, MPI_COMM_WORLD, &status);
                }
            }
        }

        // Bitonic sort
        for (int i = 0; i < (n / k); i++) {
            int master_node = i * (k / 2);
            if ((master_node <= rank) && (rank < (master_node + (k / 2)))) {
                bool ascending = (i % 2 == 0);
                bitonicSort(buf, k, master_node, rank, ascending, status);
            }
        }
        k *= 2;
    }

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

