#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>

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
#pragma omp parallel
    {
        std::vector<int> local;
        local.reserve(2048);

#pragma omp for schedule(guided, 256)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                               ? g->num_edges
                               : g->outgoing_starts[node + 1];
            int new_dist = distances[node] + 1;

            for (int e = start_edge; e < end_edge; e++)
            {
                int v = g->outgoing_edges[e];

                //  evita CAS desnecessário
                if (distances[v] == NOT_VISITED_MARKER)
                {
                    if (__sync_bool_compare_and_swap(&distances[v],
                                                     NOT_VISITED_MARKER, new_dist))
                        local.push_back(v);
                }
            }
        }

        if (!local.empty())
        {
            int offset;
#pragma omp atomic capture
            {
                offset = new_frontier->count;
                new_frontier->count += (int)local.size();
            }
            memcpy(new_frontier->vertices + offset, local.data(),
                   local.size() * sizeof(int));
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1, *new_frontier = &list2;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances);

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(Graph g, solution *sol, int depth, bool *frontier_bitmap, bool *new_frontier_bitmap, vertex_set *new_frontier)
{
#pragma omp parallel
    {
        std::vector<int> local;
        local.reserve(2048);

#pragma omp for schedule(dynamic, 1024)
        for (int v = 0; v < g->num_nodes; v++)
        {
            if (sol->distances[v] != NOT_VISITED_MARKER)
                continue;

            int start_edge = g->incoming_starts[v];
            int end_edge = (v == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[v + 1];

            for (int e = start_edge; e < end_edge; e++)
            {
                int u = g->incoming_edges[e];
                if (frontier_bitmap[u])
                {
                    sol->distances[v] = depth + 1;
                    new_frontier_bitmap[v] = true;
                    local.push_back(v);
                    break;
                }
            }
        }

        if (!local.empty())
        {
            int offset;
#pragma omp atomic capture
            {
                offset = new_frontier->count;
                new_frontier->count += (int)local.size();
            }
            memcpy(new_frontier->vertices + offset, local.data(), local.size() * sizeof(int));
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1, *new_frontier = &list2;

    bool *frontier_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bool *new_frontier_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    frontier_bitmap[ROOT_NODE_ID] = true;

    int depth = 0;

    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        memset(new_frontier_bitmap, 0, graph->num_nodes * sizeof(bool));

        bottom_up_step(graph, sol, depth, frontier_bitmap, new_frontier_bitmap, new_frontier);

        bool *tmp_bmp = frontier_bitmap;
        frontier_bitmap = new_frontier_bitmap;
        new_frontier_bitmap = tmp_bmp;

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        depth++;
    }

    free(frontier_bitmap);
    free(new_frontier_bitmap);

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

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1, *new_frontier = &list2;

    bool *frontier_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));
    bool *new_frontier_bitmap = (bool *)calloc(graph->num_nodes, sizeof(bool));

    // Init distances
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Root
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    frontier_bitmap[ROOT_NODE_ID] = true;

    int depth = 0;
    int bu_threshold = graph->num_nodes / 20;
    int td_threshold = graph->num_nodes / 25;
    if (bu_threshold < 1)
    {
        bu_threshold = 1;
    }
    if (td_threshold < 1)
    {
        td_threshold = 1;
    }

    bool use_bottom_up = false;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        if (use_bottom_up && frontier->count <= td_threshold)
            use_bottom_up = false;
        else if (!use_bottom_up && frontier->count >= bu_threshold)
            use_bottom_up = true;

        vertex_set_clear(new_frontier);
        memset(new_frontier_bitmap, 0, graph->num_nodes * sizeof(bool));

        if (use_bottom_up)
        {
            bottom_up_step(graph, sol, depth,
                           frontier_bitmap, new_frontier_bitmap,
                           new_frontier);
        }
        else
        {
            top_down_step(graph, frontier, new_frontier, sol->distances);

#pragma omp parallel for
            for (int i = 0; i < new_frontier->count; i++)
                new_frontier_bitmap[new_frontier->vertices[i]] = true;
        }

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("[Hybrid - %s] frontier=%-10d depth=%-3d  %.4f sec\n",
               use_bottom_up ? "bottom-up" : "top-down",
               frontier->count, depth, end_time - start_time);
#endif

        // Swap bitmaps and frontiers
        bool *tmp_bmp = frontier_bitmap;
        frontier_bitmap = new_frontier_bitmap;
        new_frontier_bitmap = tmp_bmp;

        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        depth++;
    }

    free(frontier_bitmap);
    free(new_frontier_bitmap);
}
