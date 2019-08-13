#include <Rcpp.h>
#include "policy.h"
using namespace std;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::DataFrame learn(Rcpp::NumericMatrix feats_R, Rcpp::NumericMatrix reward_R, int layer, int step) {

	//read data & init
	nSample = feats_R.rows();
	nFeat = feats_R.cols();
	nClass = reward_R.cols();
	level = layer;
	skip_node = step;
	//allocate();
	init();

	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nFeat; j++)
			set_feats(j,i,feats_R(i, j));
	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nClass; j++)
			set_rewards(i,j, reward_R(i, j));

	sort_feat();

	// learn the tree.
	learn_from_data(1, 1);

	// Initialize Rcpp containers and put the tree into the vectors.
	int nNode = (1 << (level + 1)) - 1;
	Rcpp::NumericVector v1(nNode);
	Rcpp::NumericVector v2(nNode);
	Rcpp::NumericVector v3(nNode);
	for(int i=0; i< nNode;i++)
	{
		v1[i] = i+1;
		v2[i] = i+1>=(1<<level)?-1:get_tree_feat(i+1);
		v3[i] = get_tree_thre(i+1);
	}

    free_memory();
	// return a dataframe object.
	return Rcpp::DataFrame::create(Rcpp::Named("node_id") = v1, Rcpp::Named("i") = v2, Rcpp::Named("b") = v3);
}


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::DataFrame learn_greedy(Rcpp::NumericMatrix feats_R, Rcpp::NumericMatrix reward_R, int layer, int step)
{
	nSample = feats_R.rows();
	nFeat = feats_R.cols();
	nClass = reward_R.cols();
	level = layer;
	skip_node = step;
	init_greedy(nSample, nFeat, nClass, level);
	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nFeat; j++)
			set_feature(i,j,feats_R(i, j));
	for (int i = 0; i < nSample; i++)
		for (int j = 0; j < nClass; j++)
			set_reward(i,j, reward_R(i, j));
	learn_greedy_1(nSample, nFeat, nClass, level);

	int nNode_greedy = (1 << (level + 1)) - 1;
	Rcpp::NumericVector v1(nNode_greedy);
	Rcpp::NumericVector v2(nNode_greedy);
	Rcpp::NumericVector v3(nNode_greedy);
	for(int i=0; i< nNode_greedy;i++)
	{
	  v1[i] = i+1;
	  v2[i] = i>=(1<<level)-1?-1:get_i_max(i);
	  v3[i] = get_b_max(i);
	}
	// return a dataframe object.
	return Rcpp::DataFrame::create(Rcpp::Named("node_id") = v1, Rcpp::Named("i") = v2, Rcpp::Named("b") = v3);
}
