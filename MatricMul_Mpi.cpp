#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
// To compile write - mpic++ A.cpp -o run  
// Then to run - mpirun -n 3 ./run  

// Function to print a matrix
void display(int rows, int cols, int *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); // init the process

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the size of the process

    int K = 10, M = 3, N = 3, P = 3;
    // if(rank == 0) {
    //     printf("Enter Number of Matrices: ");
    //     scanf("%d", &K);
    //     printf("Enter Number of Rows in Matrix A: ");
    //     scanf("%d", &M);
    //     printf("Enter Number of Columns in Matrix A: ");
    //     scanf("%d", &N);
    //     printf("Enter Number of Columns in Matrix B: ");
    //     scanf("%d", &P);
    // }

    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast from the root (rank == 0) to all processes so everyone has the same configuration:
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);// It broadcasts the value of M from the root process (rank 0) to all other processes in the communicator MPI_COMM_WORLD.
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(K % size != 0) {
        printf("Number of matrices must be divisible by the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    int A[K][M][N], B[K][N][P], R[K][M][P]; // gets allocated in the root process

    // Initialize the matrices in the root process
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;
                }
            }
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;
                }
            }
        }
    }    

    // Buffer to store portion of the matrices assigned to each process
    int localA[K / size][M][N], localB[K / size][N][P], localR[K / size][M][P]; // each processor gets K/size matrices from A and B

    // Scatter matrices to each process
    // Scatter distributes blocks of data from A and B to localA and localB:
    MPI_Scatter(A, (K / size) * M * N, MPI_INT, localA, (K / size) * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (K / size) * N * P, MPI_INT, localB, (K / size) * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // Matrix multiplication
    for(int k = 0; k < (K / size); k++) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for(int l = 0; l < N; l++) {
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;
                }
                localR[k][i][j] %= 100;
            }
        }
    }

    double endTime = MPI_Wtime();

    // Gather result matrices from all processes to the root process
    MPI_Gather(localR, (K / size) * M * P, MPI_INT, R, (K / size) * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Remove the comment to print result matrices
    //Print all the result matrices
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            printf("Result Matrix R%d\n", k);
            display(M, P, &R[k][0][0]);
        }
    }

    // Barrier to synchronize all processes before timing starts
    MPI_Barrier(MPI_COMM_WORLD);

    // Print timing information for each process
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}