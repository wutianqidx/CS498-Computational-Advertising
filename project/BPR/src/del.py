#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:13:09 2019

@author: wtq
"""
import pandas as pd
data_dir = '../../yelp_data/'.format('yelp')
social  = pd.read_csv(data_dir + "social.csv")
social_dict = {}
for  _,row in social.iterrows():
    user1,user2 = row['user1'],row['user2']
    if user1 not in social_dict:
       social_dict[user1] = [user2]
    else:
        social_dict[user1].append(user2)
print(social_dict[4553])
#for user1,user2 in social:
#    if not social_dict[user1]:
#       social_dict[user1] = [user2]
print(len(social_dict.keys()))