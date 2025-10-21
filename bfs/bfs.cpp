#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
// Versão paralela com OpenMP
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {

                // Marca o no como visitado de forma atomica
                if (__sync_bool_compare_and_swap(&distances[outgoing],
                                                 NOT_VISITED_MARKER,
                                                 distances[node] + 1))
                {

                    int index;
// Adiciona o vértice a nova fronteira de forma segura
#pragma omp atomic capture
                    index = new_frontier->count++;

                    new_frontier->vertices[index] = outgoing;
                }
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(Graph graph, solution *sol, int depth, vertex_set *new_frontier)
{
#pragma omp parallel for schedule(guided, 1024)
    for (int node = 0; node < graph->num_nodes; node++)
    {

        if (sol->distances[node] == NOT_VISITED_MARKER)
        {

            int start_edge = graph->incoming_starts[node];
            int end_edge = (node == graph->num_nodes - 1)
                               ? graph->num_edges
                               : graph->incoming_starts[node + 1];

            for (int e = start_edge; e < end_edge; e++)
            {
                int neighbor = graph->incoming_edges[e];

                if (sol->distances[neighbor] == depth)
                {
                    sol->distances[node] = depth + 1;

                    int index;
#pragma omp atomic capture
                    index = new_frontier->count++;
                    new_frontier->vertices[index] = node;
                    break;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        bottom_up_step(graph, sol, depth, new_frontier);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        depth++;
    }
}

bool should_use_bottom_up(Graph graph, vertex_set *frontier, solution *sol)
{
    double frontier_ratio = (double)frontier->count / graph->num_nodes;
    return frontier_ratio > 0.1;
}

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Init distances
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Root
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bool use_bottom_up = should_use_bottom_up(graph, frontier, sol);

        if (use_bottom_up)
            bottom_up_step(graph, sol, depth, new_frontier);
        else
            top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("[Hybrid - %s] frontier=%-10d %.4f sec\n",
               use_bottom_up ? "bottom-up" : "top-down",
               frontier->count, end_time - start_time);
#endif

        // Swaps frontiers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        depth++;
    }
}
