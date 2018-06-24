# Run arbitrary.cpp on Hydra @ULB
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
mpiCC arbitrary.cpp -o arbitrary
mpirun -np 64 ./arbitrary # Number of nodes