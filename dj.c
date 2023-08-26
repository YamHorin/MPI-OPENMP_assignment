#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <omp.h>
#include "mpi.h"

// last change 25.8 10:11

enum tags
{
    WORK,
    STOP,
    DONE_FIND_MIN_VERTEX,
    FIND_MIN_VERTEX,
    UPDATE_DISTANCES,
    MARK_VERTEX_DONE,
    DONE_UPDATE_DISTANCES,
    PRINT_DESTINATION,
    PRINT_ALL_DES
};
const int ROOT = 0;
typedef unsigned int VERTEX; //  vertices are numbered 0, 1, 2 ... (NV-1)

struct vertex
{
    VERTEX vertex;
    unsigned int distance; // distance of this vertex from vertex 0
};

const unsigned int INFINITY = 1000000; // a large integer
int reduction_result[2];
int data_pair[2];
// globals
int NV;    // number of vertices
int *done; /*  done[v] == 1 means we are done with vertex v. done[v] == 0 means we are not done yet. */
int* edges_process_own;
unsigned int *edges;
int chunk_size;
int *distance;

enum goal
{
    FIND_ONE_DISTANCE,
    FIND_ALL_DISTANCES
} goal;

VERTEX destination;

void init(int argc, char **argv);
void doWork();
struct vertex find_vertex_with_minimum_distance_WORKER(int start_place, int finish_place, int *done, int *distance);
struct vertex find_vertex_with_minimum_distance_MASTER(int num_process, MPI_Datatype mpi_vertex_type);

void update_distances_MASTER(struct vertex current, int num_process, MPI_Datatype mpi_vertex_type);
void update_distances_WORKER(struct vertex current, int start_place,
                             int finish_place, int *done, int NV, int *distance ,unsigned int* edges);
void printGraph();
void printDistances(int start , int finish);
void readGraph(void);
int main(int argc, char **argv)
{
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t_start;
    //look for an function that dies that.....
    if (num_procs < 2)
    {
        printf("eroor number of processes must be bigger then 2\n");
        MPI_Finalize();
        return 1;
    }
    struct vertex local_vertex;
    MPI_Datatype mpi_vertex_type;
    // Create an MPI data type for the struct vertex
    int block_lengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint displacements[2];
    MPI_Aint base_address;
    MPI_Get_address(&local_vertex, &base_address);
    MPI_Get_address(&local_vertex.vertex, &displacements[0]);
    MPI_Get_address(&local_vertex.distance, &displacements[1]);
    for (int i = 0; i < 2; i++)
    {
        displacements[i] -= base_address;
    }
    MPI_Type_create_struct(2, block_lengths, displacements, types, &mpi_vertex_type);
    MPI_Type_commit(&mpi_vertex_type);

    if (my_rank == 0)
    {
        init(argc, argv);
        printGraph();
        t_start = MPI_Wtime();
        int start , jump;
        int extra_vertices = NV % (num_procs - 1); // Vertices left after dividing evenly
        int remining_vertexs = NV-extra_vertices;
        /* send every process their vertexs to manage*/
        for (int worker_rank = 1; worker_rank < num_procs; worker_rank++)
        {
            MPI_Send(&NV, 1, MPI_INT, worker_rank, WORK, MPI_COMM_WORLD);
            chunk_size = (remining_vertexs /(num_procs-1));
            jump = chunk_size;
            if (extra_vertices>0 && worker_rank+1==num_procs) {
                chunk_size= chunk_size+extra_vertices;
                jump = (remining_vertexs /(num_procs-1));
            }
            
             // Adjust chunk size if there are extra vertices
            
            MPI_Send(&chunk_size, 1, MPI_INT, worker_rank, WORK, MPI_COMM_WORLD);
            MPI_Send(edges, NV*NV, MPI_UNSIGNED, worker_rank, WORK, MPI_COMM_WORLD);
            // MPI_Scatter(edges_process_own , chunk_size  , MPI_INT , NULL , 0 , MPI_INT , ROOT , MPI_COMM_WORLD); 
            start = (0 + ((worker_rank-1) * jump));
            MPI_Send(&start, 1, MPI_INT, worker_rank, WORK, MPI_COMM_WORLD);
        
        }
        struct vertex current; // current vertex and its distance from vertex 0
        for (int step = 0; step < NV; step++)
        { // step < (NV-1) should also work (see note at end of this function)
            if (step == 0)
            {
                current.vertex = 0;
                current.distance = 0;
            }
            else
                current = find_vertex_with_minimum_distance_MASTER(num_procs , mpi_vertex_type);
           
            #ifdef DEBUG
                printf("current is %u, distance is %u\n", current.vertex,
                    current.distance);
            #endif

            if (current.distance >= INFINITY)
                break;

            if (goal == FIND_ONE_DISTANCE && current.vertex == destination)
                break;      
            int send = 0; // an enum

            for (int worker_rank = 1; worker_rank < num_procs; worker_rank++)
                MPI_Send(&send, 1, MPI_INT, worker_rank, MARK_VERTEX_DONE, MPI_COMM_WORLD);
            
            for (int worker_rank = 1; worker_rank < num_procs; worker_rank++)
                MPI_Send(&current.vertex, 1, MPI_UNSIGNED, worker_rank, MARK_VERTEX_DONE, MPI_COMM_WORLD);

            update_distances_MASTER(current, num_procs , mpi_vertex_type);
        }


    }
    else /*

    worker process

    */

    {
        struct vertex local_vertex_min, current_to_update;     // for sending to the master
        int start_place, finish_place, tag, dummy; // dummy for the tags
        VERTEX done_place;
        MPI_Status status;
        MPI_Recv(&NV, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&chunk_size, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        edges = (unsigned int *)malloc(NV * NV * sizeof(unsigned int));
        distance = (int*)malloc(NV * sizeof(int));
        done =(int*) malloc(NV * sizeof(int));
        if (distance == NULL || done == NULL || edges == NULL)
        {
            fprintf(stderr, "malloc eroor ");
            exit(1);
        }
        for (VERTEX v = 0; v < NV; v++)
        {
            
            done[v] = 0;
            distance[v] = INFINITY;
        }
        distance[0] = 0;

        MPI_Recv(edges, (NV * NV), MPI_UNSIGNED, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&start_place, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        finish_place = start_place + chunk_size;
        tag = status.MPI_TAG;

        while (tag != STOP)
        {

            MPI_Recv(&dummy, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            tag = status.MPI_TAG;
            #ifdef DEBUG
                printf("rank = %d tag = %d\n",my_rank,status.MPI_TAG);
                printf("got vertex %d-%d\n\n",start_place,finish_place);

            #endif
            switch (tag)
            {
            case PRINT_ALL_DES:
                printDistances(start_place , finish_place);
            break;
            case PRINT_DESTINATION:
                MPI_Recv(&destination  , 1 , MPI_UNSIGNED , ROOT , PRINT_DESTINATION , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
                if (destination>= start_place && destination < finish_place)
                {
                    if(distance[destination] >= INFINITY)
                        printf("no path to vertex %u\n", destination);
                    else    
                        printf("%u:%u\n", destination, distance[destination]);
                }
                    
            break;
            case FIND_MIN_VERTEX:
                local_vertex_min = find_vertex_with_minimum_distance_WORKER(start_place, finish_place, done, distance);
                data_pair[0] = local_vertex_min.distance;
                data_pair[1] = local_vertex_min.vertex;
                MPI_Reduce(data_pair, reduction_result, 1, MPI_2INT, MPI_MINLOC, ROOT, MPI_COMM_WORLD);
                break;
            case UPDATE_DISTANCES:
                MPI_Recv(&current_to_update, 1, mpi_vertex_type, ROOT, UPDATE_DISTANCES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                update_distances_WORKER(current_to_update, start_place, finish_place, done, NV, distance , edges);
                break;
            case MARK_VERTEX_DONE:
                MPI_Recv(&done_place, 1, MPI_UNSIGNED, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (done_place >= start_place && done_place <= finish_place)
                {
                    #ifdef DEBUG
                        printf("done [%d] is 1\n",done_place);
                    #endif
                    done[done_place] = 1;
                }
                    
                break;
            }
        }
        

    }
    if (my_rank==0)
    {

        fprintf(stderr, "sequential time: %f secs\n", MPI_Wtime() - t_start); 
        /* time to print the results*/
        if (goal == FIND_ALL_DISTANCES)
            {
                int d=0;
                for (int i = 1; i < num_procs; i++)
                {
                    MPI_Send(&d , 1 , MPI_INT , i, PRINT_ALL_DES , MPI_COMM_WORLD);
                }
            }
	    else // goal == FIND_ONE_DISTANCE	
        {
            printf("distance from 0 to %u \n", destination);
             for (int i = 1; i < num_procs; i++)
                {
                    int d=0;
                    MPI_Send(&d , 1 , MPI_INT , i, PRINT_DESTINATION, MPI_COMM_WORLD);

                    MPI_Send(&destination , 1 , MPI_UNSIGNED , i, PRINT_DESTINATION, MPI_COMM_WORLD);

                }
        }		   
                

        /* tell every process that we done...*/
        int send = 0; // an enum

            for (int worker_rank = 1; worker_rank < num_procs; worker_rank++)
                MPI_Send(&send, 1, MPI_INT, worker_rank, STOP, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(done);
    free(distance);
    free(edges);
    return 0;
     

    
}

void init(int argc, char **argv)
{
    readGraph(); // initialize NV and 'edges'
    if (argc > 1)
    {
        goal = FIND_ONE_DISTANCE;
        destination = atoi(argv[1]);
        if (destination >= NV)
        {
            fprintf(stderr, "illegal destination vertex\n");
            exit(1);
        }
    }
    else
        goal = FIND_ALL_DISTANCES;
}


int lineno = 1; // current input line number

void skip_white_space();

void readGraph()
{

    int c;
    unsigned int w;
    int count_w = 0; // number of entries read in so far

    /* First number in the input is the number of vertices. Use it to initialize 'NV' */

    if (scanf("%d", &NV) == 1)
    {
        edges = (unsigned int *)malloc(NV * NV * sizeof(unsigned int));
        if (edges == NULL)
        {
            perror("malloc");
            exit(1);
        }
    }
    else
    {
        fprintf(stderr,
                "line %d: first item in the input should be the number of vertices in the graph\n",
                lineno);
        exit(1);
    }

    unsigned int *next_entry = edges;

    while (1)
    {
        skip_white_space();
        c = getchar();
        if (c == EOF)
            break;
        if (count_w >= NV * NV)
        {
            fprintf(stderr, "line %d: too many weights (expecting %d*%d weights)\n",
                    lineno, NV, NV);
            exit(5);
        }
        if (c == '*')
        {
            *next_entry++ = INFINITY;
            count_w++;
        }
        else
        {
            ungetc(c, stdin);
            int r = scanf("%u", &w);
            if (r == 1)
            { // a number (weight) was read
                *next_entry++ = w;
                count_w++;
                
            }
            else
            {
                fprintf(stderr, "line %d: error in input\n", lineno);
                exit(2);
            }
        }
    }
    if (count_w != NV * NV)
    {
        fprintf(stderr, "%d weights appear in the input (expected\
 %d weights because number of vertices is %d)\n",
                count_w, NV * NV, NV);
        exit(6);
    }
}

void skip_white_space()
{
    int c;
    while (1)
    {
        if ((c = getchar()) == '\n')
            lineno++;
        else if (isspace(c))
            continue;
        else if (c == EOF)
            break;
        else
        {
            ungetc(c, stdin); // push non space character back onto input stream
            break;
        }
    }
}

void printDistances(int start_place , int finish_place)
{
    int local_start = (start_place < 0) ? 0 : start_place;
    int local_finish = (finish_place > NV) ? NV : finish_place;
    for (VERTEX v = local_start; v < local_finish; v++)
        if (distance[v] >= INFINITY)
            printf("no path to vertex :%u\n", v);
        else
            printf("%u:%u\n", v, distance[v]);
}

// can be used for debugging
void printGraph()
{

    printf("graph weights:\n");
    for (int i = 0; i < NV; i++)
    {
        for (int j = 0; j < NV; j++)
            if (edges[NV * i + j] >= INFINITY)
                printf("*  ");
            else
                printf("%u  ", edges[NV * i + j]);
        putchar('\n');
    }
}

struct vertex find_vertex_with_minimum_distance_WORKER(int start_place, int finish_place, int *done, int *distance)
{
    int local_start = (start_place < 0) ? 0 : start_place;
    int local_finish = (finish_place > NV) ? NV : finish_place;

#pragma omp declare reduction(vertex_min_func : struct vertex : \
    omp_out = (omp_out.distance < omp_in.distance ? omp_out : omp_in)) \
    initializer(omp_priv = omp_orig)

    struct vertex vmin;
    vmin.distance = INFINITY;
    vmin.vertex = -1; // Initialize vertex to an invalid value

#pragma omp parallel for reduction(vertex_min_func : vmin)
    for (VERTEX v = local_start; v < local_finish; v++)
    {
        if (!done[v] && distance[v] < vmin.distance)
        {
            vmin.distance = distance[v];
            vmin.vertex = v;
        }
    }
    #ifdef DEBUG
        printf("start = %d finish  = %d\n\n", start_place , finish_place);
        printf("Final minimum vertex = %d with distance = %d\n", vmin.vertex, vmin.distance);
    #endif

    return vmin; // note: when vmin.distance is INFINITY, vmin.vertex is -1 indicating it's not found
}
struct vertex find_vertex_with_minimum_distance_MASTER(int num_process, MPI_Datatype mpi_vertex_type) // fix
{
    int send =0; // an enum
   for (int worker_rank = 1; worker_rank < num_process; worker_rank++)
                MPI_Send(&send, 1, MPI_INT, worker_rank, FIND_MIN_VERTEX, MPI_COMM_WORLD);
    
    data_pair[0] = INFINITY;
    data_pair[1] = INFINITY;//a large number for reduction 
    MPI_Reduce(data_pair, reduction_result, 1, MPI_2INT, MPI_MINLOC, ROOT, MPI_COMM_WORLD);
    struct vertex vmin;
    vmin.distance = reduction_result[0];
    vmin.vertex = reduction_result[1]; 
    return vmin;
}
void update_distances_MASTER(struct vertex current, int num_process, MPI_Datatype mpi_vertex_type)
{
    int send = 0;
    for (int i = 1; i < num_process; i++)
    {
        MPI_Send(&send , 1 , MPI_INT , i , UPDATE_DISTANCES  , MPI_COMM_WORLD);
        MPI_Send(&current , 1 , mpi_vertex_type , i , UPDATE_DISTANCES  , MPI_COMM_WORLD);
    }
}
void update_distances_WORKER(struct vertex current, int start_place, int finish_place, int *done, int NV, int *distance ,unsigned int* edges)
{
    
    int local_start = (start_place < 0) ? 0 : start_place;
    int local_finish = (finish_place > NV) ? NV: finish_place;

    #pragma omp parallel for
    for (VERTEX v = local_start; v < local_finish; v++)
    {
        if (!done[v])
        {
            unsigned int alternative = current.distance + edges[current.vertex * NV + v];
            // #ifdef DEBUG
            //     printf("alternative: %u for VERTEX %u\n", alternative, v);
            // #endif
            if (alternative < distance[v])
                *(distance + v) = alternative;
        }
    }
        
}


