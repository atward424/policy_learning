#ifndef _POLICY_H_
#define _POLICY_H_

#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <bitset>
#include <vector>
#include <map>
#include <ctime>

#define NO_MAP 0
#define HASHING 2
using namespace std;

extern int mapping;

const double INF = 1e18 + 10;
const int MAXN = 1e6 + 10;
const int MAXNODE = (1 << 10) + 100;
extern int MAXF;
extern int MAXK;
extern int MAXHASH;

extern int nSample, nFeat, nClass, level, skip_node;
extern int cur_feat, total;
extern int **orderFeat;
extern bool *jump_leaf;
extern double **sum_reward, *leaf_thre;
extern double **feats, **rewards, best_reward;

struct treeNode
{
    int action, feat;
    double threshold;
    bitset<MAXN> vis;
};
extern treeNode tree[];

struct hashNode
{
    bitset<MAXN> vis;
    int level;
    double reward;
};
extern vector<hashNode> *hash_Nodes;

unsigned int hashing(bitset<MAXN> &, int);
bool equal(bitset<MAXN> &, bitset<MAXN> &, int);
double search_hash(bitset<MAXN> &, int, unsigned int);

//interface start
void set_feats(int, int, double);
void set_rewards(int, int, double);
int get_tree_feat(int);
double get_tree_thre(int);
//end

void sort_feat();
void init();
double leaf_learning(int, unsigned int);
double learn_from_data(int, int, bool = true);

template <typename T>
T *allocate(int);
template <typename T>
T **allocate(int, int);
template <typename T>
void delete_array(T*);
template <typename T>
void delete_array(T**, int);
void free_memory();





//greedy
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <utility>
#include <cmath>
#include <ctime>
#include<fstream>
#include <math.h>
extern int n, p, k, l, nodenumber, id;
extern int *i_max;
extern double *b_max;
extern int *bb_max;
extern int **label_save;
extern double *reward_max;
extern int learn_greedy_1(int nSample, int nFeat, int nClass, int level);
extern int test_gr(int testnumber);
extern void greedy_test(int nSample, int nFeat, int nClass, int level);
extern void set_feature(int i, int j, double feature_in);
extern void set_reward(int i, int j, double reward_in);
extern void init_greedy(int nSample, int nFeat, int nClass, int level);
extern int get_i_max(int idnow);
extern double get_b_max(int idnow);


#endif
