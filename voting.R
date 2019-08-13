setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("./pipeline/install_packages.R")
source("./pipeline/estimator_functions.R")
sourceCpp('./pipeline/Rport.cpp')
source("./pipeline/evaluation.R")
source("./pipeline/tree_visualization.R")
options(scipen = 100)

calc_gamma = FALSE

## Set parameters.
level_of_tree <- 2  # decide the level of learnt tree
jump_step <- 1  # step length


################################
##### Step 1: read in data #####
################################

training.data = read.csv("./clean_data/voting_data.csv")
n = nrow(training.data)
p = ncol(training.data) - 2
features = training.data[, 1:p]
actions = training.data[, p + 1]
rewards = training.data[, p + 2]
k = length(unique(actions))

# select which features to use
features = as.matrix(features[, c(2:8,14, 60)])

# since this was a randomized control trial, probabilities of each action are known
probs = matrix(rep(c(10/18, 2/18, 2/18, 2/18, 2/18), nrow(features)), ncol = 5, byrow = T)

# print out which features are included in the data
colnames(features)

insert_interactions = FALSE
if (insert_interactions) {
  prob_flip = 0.07
  age_less50_small_household = c(0, -1, 1, -1, -1)*prob_flip
  age_less50_big_household = c(0, -1, -1, 1, -1)*prob_flip
  age_over50_medium_household = c(0, 1, -1, -1, -1)*prob_flip
  
  ages = 2006 - training.data$yob
  size = training.data$hh_size
  
  act_cor = actions + 1
  for (i in 1:nrow(training.data)) {
    if (act_cor[i] > 1) {
      if (ages[i] < 50 && size[i] < 3) {
        flip_val = age_less50_small_household[act_cor[i]]
        if (runif(1) < abs(flip_val)) {
          rewards[i] = rewards[i] + flip_val/abs(flip_val)
        }
      }
      if (ages[i] < 50 && size[i] >= 3) {
        flip_val = age_less50_big_household[act_cor[i]]
        if (runif(1) < abs(flip_val)) {
          rewards[i] = rewards[i] + flip_val/abs(flip_val)
        }
      }
      if (ages[i] >= 50 && size[i] > 1) {
        flip_val = age_over50_medium_household[act_cor[i]]
        if (runif(1) < abs(flip_val)) {
          rewards[i] = rewards[i] + flip_val/abs(flip_val)
        }
      }
    }
  }
  
  rewards[rewards > 1] = 1
  rewards[rewards < 0] = 0
  
  table(rewards)
}


################################
### Step 2  Run estimation #####
################################

if (exists('actions_all_folds')) {
  rm(actions_all_folds)
}
if (exists('gamma_all_folds')) {
  rm(gamma_all_folds)
}

rand_inds = sample(n, n)
method_names = c('DR_tree', 'DR_direct_tree', 'IPS_tree', 'CRF_tree', 'random')
act_names = c('nothing', 'civic', 'monitored', 'self_history', 'neighbors')
act_names = paste0('a_', act_names)

# k-fold cross validation
folds = 5
for (fold in 1:folds) {
  
  # split current fold into training and test samples
  test_inds = rand_inds[(ceiling((fold - 1)*n/folds)+1):(ceiling((fold)*n/folds))]
  train_inds = rand_inds[!(rand_inds %in% test_inds)]#[1:1000]
  
  
  Gammas = calculate.all.Gammas(features[train_inds, ], rewards[train_inds], actions[train_inds], probs[train_inds,  ],
                                test_features = features[test_inds, ], test_actions=actions[test_inds], test_rewards=rewards[test_inds],
                                probs[test_inds, ], k = k, train_prob_type = 'est', test_prob_type = 'exact', prob_clip = 0.1, direct = TRUE)
  
  est_rewards_DR = Gammas$AIPW_causal
  est_rewards_IPS = Gammas$IPW
  est_rewards_CFR = Gammas$DM_causal
  test_rewards_CFR = Gammas$DM_causal_test
  est_rewards_DM = Gammas$DM_direct
  est_rewards_DR_direct = Gammas$AIPW_direct
  
  test_rewards_DR = Gammas$AIPW_eval
  
  # alternatively, we could estimate each set of gammas separately as follows (but this will take longer)
  
  # est_rewards_DR_list = calculate.Gamma.with.test(features[train_inds, ], rewards[train_inds], actions[train_inds], k, 
  #                                                 prob_estimation = 'rf', mu_estimation = 'rf', method = 'DR', exact_probs = probs[train_inds, ],
  #                                                 test_features = features[test_inds, ], test_actions=actions[test_inds], test_rewards=rewards[test_inds], 
  #                                                 test_exact_probs = probs[test_inds, ], test_prob_type = 'exact', prob_clip = 1/6, 
  #                                                 forest_type = 'direct')
  # 
  # 
  # est_rewards_DR = est_rewards_DR_list$train
  # test_rewards_DR = est_rewards_DR_list$test

  # est_rewards_IPS = calculate.Gamma(features[train_inds, ], rewards[train_inds], actions[train_inds], k, 
  #                                   prob_estimation = 'exact', mu_estimation = 'rf', method = 'IPS', exact_probs = probs[train_inds, ])
  
  # est_rewards_CFR_list = calculate.Gamma.with.test(features[train_inds, ], rewards[train_inds], actions[train_inds], k, 
  #                                                   prob_estimation = 'rf', mu_estimation = 'rf', method = 'CFR', exact_probs = probs[train_inds, ],
  #                                                   test_features = features[test_inds, ], test_actions=actions[test_inds], test_rewards=rewards[test_inds], 
  #                                                   test_exact_probs = probs[test_inds, ], test_prob_type = 'est', prob_clip = 1/6, 
  #                                                  forest_type = 'causal')
  # est_rewards_CFR = est_rewards_CFR_list$train
  # test_rewards_CFR = est_rewards_CFR_list$test
  
  # print('est_rewards_DR avgs:')
  # print(apply(est_rewards_DR, 2, mean))
  # print('test_rewards_DR avgs:')
  # print(apply(test_rewards_DR, 2, mean))
  # print('est_rewards_IPS avgs:')
  # print(apply(est_rewards_IPS, 2, mean))
  # print('est_rewards_CFR avgs:')
  # print(apply(est_rewards_CFR, 2, mean))
  # print('test_rewards_CFR avgs:')
  # print(apply(test_rewards_CFR, 2, mean))
  
  
  
  ################################
  ##### Step 3 - Learn trees #####
  ################################
  
  tree_learnt_DR =  learn(features[train_inds, ], est_rewards_DR, level_of_tree, jump_step)
  tree_learnt_DR_direct =  learn(features[train_inds, ], est_rewards_DR_direct, level_of_tree, jump_step)
  tree_learnt_IPS = learn(features[train_inds, ], est_rewards_IPS, level_of_tree, jump_step)
  # tree_learnt_DFRT =  learn(features[train_inds, ], rf_regress_gamma_train, level_of_tree, jump_step)
  # tree_learnt_DR_greedy = learn_greedy(features[train_inds, ], est_rewards_DR, level_of_tree, jump_step)
  tree_learnt_CFR =  learn(features[train_inds, ], est_rewards_CFR, level_of_tree, jump_step)
  
  visualize(tree_learnt_DR, colnames(features), act_names, paste('voting', fold, "DR_nonnormalized.png",  sep = '_'), TRUE)
  visualize(tree_learnt_IPS, colnames(features), act_names, paste('voting', fold, "IPS_nonnormalized.png", sep = '_'), TRUE)
  
  
  ################################
  #####  Step 4 - Evaluate   #####
  ################################
  
  a_DR_tree = classify.via.tree(features[test_inds, ], tree_learnt_DR)+1
  a_DR_direct_tree = classify.via.tree(features[test_inds, ], tree_learnt_DR_direct)+1
  a_IPS_tree = classify.via.tree(features[test_inds, ], tree_learnt_IPS)+1
  a_CFR_tree = classify.via.tree(features[test_inds, ], tree_learnt_CFR)+1
  const_actions = matrix(rep(1:length(act_names), length(test_inds)), nrow = length(test_inds), byrow = T)
  rand_actions = sample(k, length(test_inds), replace = T)
  
  all_actions = data.frame(a_DR_tree, a_DR_direct_tree, a_IPS_tree, a_CFR_tree, rand_actions, const_actions)
  colnames(all_actions) = c(method_names, act_names)
  
  results = get.results.from.actions(all_actions, test_rewards_DR)
  
  # print(results$means)
  
  if (exists("actions_all_folds")) {
    actions_all_folds = rbind(actions_all_folds, all_actions)
  } else {
    actions_all_folds = all_actions
  }
  
  if (exists("gamma_all_folds")) {
    gamma_all_folds = rbind(gamma_all_folds, test_rewards_DR)
  } else {
    gamma_all_folds = test_rewards_DR
  }
  
}

results_all = get.results.from.actions(actions_all_folds, gamma_all_folds)

print(results_all$means)

print(results_all$diffs[, 1:length(method_names)])
print(results_all$p_vals[, 1:length(method_names)])
