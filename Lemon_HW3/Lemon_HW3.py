# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:20:46 2019

@author: lemon
"""

#%% Function
    
def gen_senti(string):
    import pandas as pd
    import re
    from nltk import word_tokenize

    nw = pd.read_csv("/posNeg/negative-words.txt", header = None, encoding = "ISO-8859-1", names = ["neg"])
    pw = pd.read_csv("/posNeg/positive-words.txt", header = None, encoding = "ISO-8859-1", names = ["pos"])
    nw = nw["neg"].values.tolist()
    pw = pw["pos"].values.tolist()
    
    nc = 0
    pc = 0
    
    clean = re.sub('[^A-Za-z]+', " ", string)
    clean = clean.lower()

    for word in word_tokenize(clean):
        if word in nw:
            nc -= 1
        elif word in pw:
            pc += 1

    if pc == 0 and nc == 0:
        return("No positive or negative words found.")
    else:
        S = (nc + pc)/(abs(nc) + pc)
        return(S)
    
#%% Test
    
gen_senti("Achievements abound in this classroom, however these test scores are an absurd, abysmal abomination.")

# pos words: achievements, abound
# neg words: absurd, abysmal, abomination
# -1 + -1 + -1 + 1 + 1/5 = -0.2

#%% For Q3

gen_senti("The darkest hour is among us in this time of gloom, however, we will prevail!")

#%% Answers to part 2
#
# 1. iterate_var is a function that takes target_var as the argument. It takes a feature
# matrix in tfidf and reduces the number of dimensions (my_dim) to the point that 
# the explained variance (var_fig) is at or higher than target_var. The variable pca
# uses Principal Component Analysis to reduce the dimensions, where cnt is equal to
# the number of dimensions. cnt starts at 1, meaning 1 dimension. As cnt increases, 
# the number of dimensions increases, leading to generally a higher var_fig. Once var_fig
# is equal to or greater than target_var, the while loop terminates and the function returns
# a dataframe of the principal components. In this case, this function reduced the number
# of features/components/dimensions from 16,230 to 134 while still explaining %95 of the variability.
#
# a. explained variance when dimensions = 1
# b. if explained variance < target: dimensions + 1
# c. keep going through while loop till explained variance >= target
# d. return dataframe feature matrix with dimension reduction (PCA)
#
# 2. grid_search_func is a function that takes param_grid, the_mode_in, the_vec_in, 
# and the_lab_in as the arguments. GridSearchCV takes a grid of hyperparameters 
# (param_grid) and then produces a series of models from all possible combinations of hyperparameters. 
# In this case the hyperparameters are max_depth and n_estimators. Then, 
# the function searches for the optimal model that can be produced from the hyperparameters.
# The function returns the best possible model (best_model), the score of the best model (max_score),
# and the parameters of the best model (best_params). In my run, the max_score was 0.61988.
# 
# a. Pass grid (param_grid)
# b. Pass model in (RandomForestClassifier)
# c. Cycle through all combinations of parameters
# d. Fit each model
# e. Test and validate 
# f. pull best score
# e. outputs the optimal prediction
#
# 3. -1.0
#
# 4. negative_words = 0.541667, positive_words = 0.458333               
#
# 5. A trigram includes negations (not nobody) and double negations (noone, not nobody)
# which will add more context to the words and therefore change the scores, likely
# skewing the scores more to negative.
# negative_words = 0.692187, positive_words = 0.307812             
#