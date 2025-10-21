#ifndef __BFS_H__
#define __BFS_H__

// #define DEBUG

#include "common/graph.h"

struct solution
{
  int *distances;
};

struct vertex_set
{
  // # of vertices in the set
  int count;
  // max size of buffer vertices
  int max_vertices;
  // array of vertex ids in set
  int *vertices;
};

void bfs_top_down(Graph graph, solution *sol);
void bfs_bottom_up(Graph graph, solution *sol);
void bottom_up_step(Graph graph, solution *sol, int depth, vertex_set *new_frontier);
void bfs_hybrid(Graph graph, solution *sol);

#endif
