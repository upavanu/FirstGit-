# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:15:51 2019

@author: PAVAN
"""

from recommender import data,algo
trainingSet = data.build_full_trainset()
algo.fit(trainingSet)
prediction=algo.predict('E',2)
prediction.est