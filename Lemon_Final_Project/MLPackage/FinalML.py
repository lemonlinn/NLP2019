# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:40:51 2019

@author: Lemon Lin Reimer
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

class Jesse(object):
    
    def __init__(self):
        print("Let's get this started!")
    
    def iterate_var(self, my_xform_tfidf_in, var_target, data_slice):
        """PCA using smallest possible # of components to hit target EVR"""
        print("Dimension reduction in progress! One moment, please!")
        var_fig = 0.0
        cnt = 1
        while var_fig <= var_target:
            pca = PCA(n_components=cnt)
            my_dim = pca.fit_transform(my_xform_tfidf_in[0:data_slice])
            var_fig = sum(pca.explained_variance_ratio_)   
            cnt += 1
        pca = PCA(n_components=cnt)
        my_dim = pca.fit_transform(my_xform_tfidf_in)

        return(my_dim, pca)
    
    def grid_search(self, param_grid, the_mode_in, the_vec_in, the_lab_in):
        """Gives best possible combination of hyperparameters based on grid options"""
        print("Grid search in progress! One moment, please!")
        grid_search = GridSearchCV(the_mode_in, param_grid=param_grid, cv=5)
        best_model = grid_search.fit(the_vec_in, the_lab_in)
        max_score = grid_search.best_score_
        best_params = grid_search.best_params_

        return(best_model, max_score, best_params)
    
    def prediction(self, data_in, data_out, my_vec_tfidf, pca, clf_pca):
        """Builds DF of predictions based on pre-labeled test data"""
        print("Prediction in progress! One moment, please!")
        the_predicted_out = pd.DataFrame()
        for word, actual in zip(data_in,data_out):
            test_text = my_vec_tfidf.transform([word]).toarray()
            test_text_pca = pca.transform(test_text)
            the_probs = list(clf_pca.predict_proba(test_text_pca)[0])
            the_predict = clf_pca.predict(test_text_pca)[0]
            the_probs.append(the_predict)
            the_probs.append(actual)
            the_result = pd.DataFrame(the_probs).T
            tmp_cols =np.append(clf_pca.classes_, 'predicted') 
            the_result.columns = np.append(tmp_cols, 'actual')
            the_predicted_out = the_predicted_out.append(the_result, ignore_index=True)

        return(the_predicted_out)

    