#include "policy.h"

//config start
int mapping = 0;
int MAXMEM = 7;// max memory of PC. 7GB default.
//end

int MAXF = 80;// max feature
int MAXK = 16;// max class
int MAXHASH = 1e6 + 100;

int nSample, nFeat, nClass, level, skip_node = 1;
int cur_feat, total;
int **orderFeat;
bool *jump_leaf;
double **sum_reward, *leaf_thre;
double **feats, **rewards, best_reward;

//hash start
int cur_hashnode = 0;
int maxn_hashnode = (int)(MAXMEM*1e9 / (MAXN / 8.0 + 13));
treeNode tree[MAXNODE];
vector<hashNode> *hash_nodes;
unsigned int hashing(bitset<MAXN> &vis, int level)
{
	unsigned int hash = 1;
	int pos = 0;
	while (pos < nSample)
	{
		unsigned int tem = 0;
		for (int i = 0; i < 32; i++)
		{
			tem <<= 1;
			tem |= vis[pos];
			pos++;
			if (pos == nSample)
				break;
		}
		if (tem != 0)
		{
			hash *= (tem | 1);
		}
	}
	hash = hash * level * level * level;
	return hash % MAXHASH;
}
bool equal(bitset<MAXN> &a, bitset<MAXN> &b, int n)
{
	for (int i = 0; i < n; i++)
		if (a[i] ^ b[i])
			return 0;
	return 1;
}
double search_hash(bitset<MAXN> &vis, int level, unsigned int hash_code)
{
	for (unsigned int i = 0; i < hash_nodes[hash_code].size(); i++)
	{
		hashNode tem = hash_nodes[hash_code][i];
		if (equal(tem.vis, vis, nSample) && tem.level == level)
		{
			return tem.reward;
		}
	}
	return -INF;
}
inline void add_hash(bitset<MAXN> &vis, int level, double reward, int hash_code)
{
	if (++cur_hashnode > maxn_hashnode) return;
	hash_nodes[hash_code].push_back({vis, level, reward});
}
//end

//interface start
void set_feats(int j, int i, double val)
{
	feats[j][i] = val;
}
void set_rewards(int i, int j, double val)
{
	rewards[i][j] = val;
}
int get_tree_feat(int i)
{
	return (int)tree[i].feat;
}
double get_tree_thre(int i)
{
	return tree[i].threshold;
}
//end

int cmp(int a, int b)
{
	return feats[cur_feat][a] > feats[cur_feat][b];
}
void sort_feat()
{
	for (int i = 0; i < nFeat; i++)
	{
		cur_feat = i;
		sort(orderFeat[i], orderFeat[i] + nSample, cmp);
	}
}
template <typename T>
T *allocate(int m)
{
	T* a = new T[m]();
	return a;
}
template <typename T>
T **allocate(int m, int n)
{
	T **a = new T *[m];
	for (int i = 0; i < m; i++)
	{
		a[i] = new T[n]();
	}
	return a;
}

template <typename T>
void delete_array(T* a)
{
    delete []a;
}
template <typename T>
void delete_array(T** a,int m)
{
    for(int i=0;i<m;i++)
        delete[]a[i];
}

void init()
{
	jump_leaf = allocate<bool>(MAXN);
	orderFeat = allocate<int>(MAXF, MAXN);
	sum_reward = allocate<double>(MAXN, MAXK);
	leaf_thre = allocate<double>(MAXN);
	feats = allocate<double>(MAXF, MAXN);
	rewards = allocate<double>(MAXN, MAXK);
	hash_nodes = allocate<vector<hashNode>>(MAXHASH);

	cur_hashnode = 0;
	for (int i = 0; i < MAXHASH; i++)
		hash_nodes[i].clear();
	memset(tree, 0, sizeof tree);
	for (int i = 0; i < nSample; i++)
		tree[1].vis.set(i);
	for (int i = 0; i < nFeat; i++)
		for (int j = 0; j < nSample; j++)
			orderFeat[i][j] = j;
}

/*void free_memory()
{
    //free memory
    delete [] jump_leaf;
    delete [] orderFeat;
    delete [] sum_reward;
    delete [] leaf_thre;
    delete [] feats;
    delete [] rewards;
    delete [] hash_nodes;
}*/

void free_memory()
{
    delete_array(jump_leaf);
    delete_array(orderFeat, MAXF);
    delete_array(sum_reward, MAXN);
    delete_array(leaf_thre);
    delete_array(feats, MAXF);
    delete_array(rewards, MAXN);
    delete_array(hash_nodes);
}

double leaf_learning(int i, unsigned int hash_code)
{
	double best_reward = -INF;
	// left child and right child
	int left = i << 1, right = (i << 1) + 1;
	for (int k = 0; k < nFeat; k++)
	{
		int count = 1, total = 0, cur_skip = 0;
		//find the reward matrix
		for (int j = 0; j < nSample;)
		{
			// from biggest feature to smallest
			int pos = orderFeat[k][j], new_pos = 0;
			j++;
			if (!tree[i].vis[pos])
				continue;
			for (int l = 0; l < nClass; l++)
				sum_reward[count][l] = rewards[pos][l] + sum_reward[count - 1][l];
			total++;
			// adding the node with same feature
			while (j < nSample)
			{
				new_pos = orderFeat[k][j];
				while (j < nSample && !tree[i].vis[new_pos])
				{
					j++;
					new_pos = orderFeat[k][j];
				}
				if (j == nSample || feats[k][pos] != feats[k][new_pos])
					break;
				total++;
				for (int l = 0; l < nClass; l++)
					sum_reward[count][l] = rewards[new_pos][l] + sum_reward[count][l];
				j++;
			}
			// leaf_thre[i] is the threshold when cutting between i-1 and i
			leaf_thre[count] = (feats[k][pos] + (j == nSample ? 0 : feats[k][new_pos])) / 2;
			// to decide jumping at node count
			jump_leaf[count] = total / skip_node >= cur_skip;
			if (jump_leaf[count])
				cur_skip++;
			count++;
		}
		// calculate the best reward
		for (int j = 0; j < count; j++)
		{
			if (!jump_leaf[j])
				continue;
			double up_best = -INF, dw_best = -INF;
			double left_action = 0, right_action = 0;
			for (int l = 0; l < nClass; l++)
			{
				if (up_best < sum_reward[j][l])
				{
					up_best = sum_reward[j][l];
					right_action = l;
				}
				if (dw_best < sum_reward[count - 1][l] - sum_reward[j][l])
				{
					dw_best = sum_reward[count - 1][l] - sum_reward[j][l];
					left_action = l;
				}
			}
			if (best_reward < up_best + dw_best)
			{
				best_reward = up_best + dw_best;
				tree[i].threshold = j ? leaf_thre[j] : feats[k][orderFeat[k][0]] + 1;
				tree[i].feat = k;
				tree[left].action = tree[left].threshold = left_action;
				tree[right].action = tree[right].threshold = right_action;
			}
		}
	}
	// if there is no node, set everything to -1
	if (best_reward == -INF)
	{
		best_reward = 0;
		tree[i].threshold = -1;
		tree[i].feat = -1;
		tree[left].action = tree[left].threshold = -1;
		tree[right].action = tree[right].threshold = -1;
	}
	if (mapping == 2)
		add_hash(tree[i].vis, 1, best_reward, hash_code);
	return best_reward;
}
double learn_from_data(int i, int layer, bool memory)
{
	pair<bitset<MAXN>, int> key;
	unsigned int hash_code = 0;
	if (memory && mapping == 2)
	{
		hash_code = hashing(tree[i].vis, level - layer + 1);
		double reward = search_hash(tree[i].vis, level - layer + 1, hash_code);
		if (reward != -INF)
			return reward;
	}
	// father of leaf node
	if (layer >= level)
		return leaf_learning(i, hash_code);

	double cur_best = -INF, best_feat = -1, best_thre = -1;
	treeNode best_left, best_right;
	// left child, right child
	int left = i << 1, right = left + 1;
	for (int k = 0; k < nFeat; k++)
	{
		int cur_skip = 0, total = 0;
		tree[left].vis = tree[i].vis;
		tree[right].vis.reset();
		double a = learn_from_data(left, layer + 1);
		double b = learn_from_data(right, layer + 1);
		if (cur_best < a + b)
		{
			cur_best = a + b;
			best_feat = k;
			best_left = tree[left];
			best_right = tree[right];
			best_thre = feats[k][orderFeat[k][0]] + 1;
		}
		// separate the nodes from big feature node to samll one
		for (int j = 0; j < nSample;)
		{
			int pos = orderFeat[k][j], new_pos = 0;
			j++;
			if (!tree[left].vis[pos])
				continue;
			tree[left].vis.reset(pos);
			tree[right].vis.set(pos);
			total++;
			// jump the node with same feature
			while (j < nSample)
			{
				new_pos = orderFeat[k][j];
				while (j < nSample && !tree[left].vis[new_pos])
				{
					j++;
					new_pos = orderFeat[k][j];
				}
				if (j == nSample || feats[k][pos] != feats[k][new_pos])
					break;

				tree[left].vis.reset(new_pos);
				tree[right].vis.set(new_pos);
				total++;
				j++;
			}
			// jump node
			if (cur_skip > total / skip_node && j != nSample)
				continue;

			a = j == nSample ? 0 : learn_from_data(left, layer + 1);
			b = learn_from_data(right, layer + 1);

			cur_skip++;
			if (a + b > cur_best)
			{
				cur_best = a + b;
				best_feat = k;
				best_thre = j == nSample ? feats[k][pos] : ((feats[k][pos] + feats[k][new_pos]) / 2);
				best_left = tree[left];
				best_right = tree[right];
			}
		}
	}

	if (cur_best == -INF)
		cur_best = 0;
	tree[i].feat = best_feat;
	tree[i].threshold = best_thre;
	tree[left] = best_left;
	tree[right] = best_right;
	learn_from_data(left, layer + 1, false);
	learn_from_data(right, layer + 1, false);

	if (mapping == 2)
		add_hash(tree[i].vis, level - layer + 1, cur_best, hash_code);
	return cur_best;
}




//greedy factor_zh;
int n = 0, p = 0, k = 0, l = 0, nodenumber = 0, id = 0;
int *i_max;
double **feature;
double **reward;
double *b_max;
int *bb_max;
int **label_save;
double *reward_max;
char feat_path[100], reward_path[100];
char test_path[100];
//greedy_learn_zh
void find_i_b(double rewardmax, double **sum_reward, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (rewardmax == sum_reward[i][j])
			{
				i_max[id] = i;
				bb_max[id] = j;
				return;
			}
		}
	}
}
int length(int*label, int n)
{
	int number = 0;
	for (int i = 0; i < n; i++)
	{
		if (label[i] == 1)
			number++;
	}
	return number;
}

double max1(double *arr, int number)
{
	double temp = arr[0];
	for (int i = 1; i<number; i++)
	{

		if (temp<arr[i])
		{
			temp = arr[i];
		}
	}
	return temp;
}

double max2(double **arr, int rows, int cols)
{
	double temp = arr[0][0];
	for (int i = 0; i<rows; i++)
		for (int j = 0; j<cols; j++)
			if (temp <= arr[i][j])
			{
				temp = arr[i][j];
			}
	return temp;
}
void leaf_learning(int *label, double **feature, double **reward)
{
	int bb, j, i = 0;
	double b = 0;
	int number = length(label, n);
	double*feature_new = new double[number];
	double*sum_reward_left = new double[k];
	double*sum_reward_right = new double[k];
	for (i = 0; i < k; i++)
		sum_reward_left[i] = 0;
	for (i = 0; i < k; i++)
		sum_reward_right[i] = 0;
	double **sum_reward;
	sum_reward = new double*[p];
	for (i = 0; i<p; i++)
		sum_reward[i] = new double[number];
	for (i = 0; i < p; i++)
	{
		j = 0;
		for (int count = 0; count < n; count++)
		{
			if (label[count] == 1)
			{
				feature_new[j] = feature[count][i];
				j++;
			}
		}
		j = 0;
		sort(feature_new, feature_new + number);
		if (id >= pow(2, l) - 1)
			i_max[id] = -1;
		if (number == 0)
		{
			b_max[id] = -66666;
			i_max[id]=-1;
			return;
		}
		for (bb = 0; bb < number; bb++)
		{
			b = feature_new[bb];
			if (id >= pow(2, l) - 1)
			{
				b = feature_new[number - 1];
			}
			for (int j = 0; j < n; j++)
			{
				if (label[j] == 1)
				{
					if (feature[j][i] <= b)
					{
						for (int kk = 0; kk < k; kk++)
						{
							sum_reward_left[kk] = sum_reward_left[kk] + reward[j][kk];
						}
					}
					else
					{
						for (int kk = 0; kk < k; kk++)
						{
							sum_reward_right[kk] = sum_reward_right[kk] + reward[j][kk];
						}
					}
				}
			}
			if (id >= pow(2, l) - 1)
			{
				double temp = 0;

				for (int m = 0; m < k; m++)
				{
					if (temp <= sum_reward_left[m])
					{
						temp = sum_reward_left[m];
						b_max[id] = m;
					}
				}
			}

			sort(sum_reward_right, sum_reward_right + k);
			sort(sum_reward_left, sum_reward_left + k);
			sum_reward[i][bb] = sum_reward_right[k - 1] + sum_reward_left[k - 1];
			for (int x = 0; x < k; x++)
				sum_reward_left[x] = 0;
			for (int x = 0; x < k; x++)
				sum_reward_right[x] = 0;


		}
	}

	reward_max[id] = max2(sum_reward, p, number);
	if (id < pow(2, l) - 1)
	{
		find_i_b(reward_max[id], sum_reward, p, number);
		j = 0;
		for (int count = 0; count < n; count++)
		{
			if (label[count] == 1)
			{
				feature_new[j] = feature[count][i_max[id]];
				j++;
			}
		}
		sort(feature_new, feature_new + number);
		b_max[id] = feature_new[bb_max[id]];
	}
}
void fenlei(int *label, int *i_max, double *b_max, double **feature)
{
	for (int i = 0; i < n; i++)
	{
		if (label[i] == 1)
		{
			if (feature[i][i_max[id]] <= b_max[id])
			{
				label_save[2 * id + 1][i] = 1;
			}
			else
			{
				label_save[2 * id + 2][i] = 1;
			}
		}

	}
	return;
}
int learn_greedy_1(int nSample, int nFeat, int nClass, int level)
{
	int i;
	double reward_all = 0;
	int *label_in = new int[n];
	for (id = 0; id <pow(2, l + 1) - 1; id++)
	{
		for (i = 0; i < n; i++)
		{
			label_in[i] = label_save[id][i];
		}

		leaf_learning(label_in, feature, reward);
		if (id<pow(2, l) - 1)
			fenlei(label_in, i_max, b_max, feature);

	}
	for (i = 0; i < pow(2,l); i++)
	{
		reward_all = reward_all + reward_max[nodenumber - i - 1];
	}
	return nodenumber;
}

void init_greedy(int nSample, int nFeat,int nClass, int level)
{
	int i,j = 0;
	n = nSample;
	p = nFeat;
	k = nClass;
	l = level;
	nodenumber = pow(2, l + 1) - 1;
	i_max = new int[nodenumber];
	b_max = new double[nodenumber];
	bb_max = new int[nodenumber];
	reward_max = new double[nodenumber];
			for (i = 0; i < nodenumber; i++)
		{
			reward_max[i] = 0;
		}
		feature = new double*[n];
	for (i = 0; i<n; i++)
		feature[i] = new double[p];

		reward = new double*[n];
	for (i = 0; i<n; i++)
		reward[i] = new double[k];

	label_save = new int*[nodenumber];
	for (i = 0; i<nodenumber; i++)
		label_save[i] = new int[n];

	for (j = 0; j < nodenumber; j++)
		for (i = 0; i < n; i++)
			label_save[j][i] = 0;

	for (i = 0; i < n; i++)
		label_save[0][i] = 1;

}
void set_feature(int i,int j, double feature_in)
{
feature[i][j]=feature_in;
	return ;
}

void set_reward(int i,int j, double reward_in)
{
	reward[i][j]=reward_in;
	return ;
}
int get_i_max(int idnow)
{
	return i_max[idnow];
 }
 double get_b_max(int idnow)
{
	return b_max[idnow] ;
}
