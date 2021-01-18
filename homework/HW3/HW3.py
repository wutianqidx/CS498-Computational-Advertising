#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import copy


# In[2]:


# get inputs

file = 'sample_input.txt'
file = 'input_b.txt'
C = 0.8
iterations = 10

with open (file) as f:
    next(f)
    edges = []
    users = []
    ads = []
    for lines in f:
        edge = lines.strip().split(',')[0:2]
        edges.append(edge)
        users.append(edge[0])
        ads.append(edge[1])
    
    users = list(set(users))
    ads = list(set(ads))
    Q_user = edges[-1][0]
    Q_ad = edges[-1][1]
    edges = edges[:-1]
    f.close()
    
sim_user = {}
sim_ad = {}
for u in users:
    sim_user[u] = {}
    for u1 in users:
        if u==u1:
            sim_user[u][u1] = 1
        else:
            sim_user[u][u1] = 0

for a in ads:
    sim_ad[a] = {}
    for a1 in ads:
        if a==a1:
            sim_ad[a][a1] = 1
        else:
            sim_ad[a][a1] = 0

user_ad = {u:[] for u in users}
ad_user = {a:[] for a in ads}
for edge in edges:
    user_ad[edge[0]].append(edge[1])
    ad_user[edge[1]].append(edge[0])


# In[3]:


def get_item_sim(u1,u2,sim_ad,partial_ad,graph):
    sim = 0
    u1_ads = graph[u1]
    u2_ads = graph[u2]
    for u1_ad in u1_ads:
        if u2 in partial_ad[u1_ad]:
            sim += partial_ad[u1_ad][u2]
        else:
            add_partial = 0
            for u2_ad in u2_ads:
                add_partial += sim_ad[u1_ad][u2_ad]
                
            partial_ad[u1_ad][u2] = add_partial
            sim+=add_partial
    sim = sim*C/(len(u1_ads)*len(u2_ads))
    return sim,partial_ad


# In[4]:


# start iteration for simple simrank

i = 0 
while i < iterations:
    partial_ad = {a:{} for a in ads}
    partial_user = {u:{} for u in users}
    for u1 in users:
        for u2 in users:
            if u1 == u2:
                sim_user[u1][u2] = 1
            else:
                sim_user[u1][u2],partial_ad = get_item_sim(u1,u2,sim_ad,partial_ad,user_ad)
                    
    for a1 in ads:
        for a2 in ads:
            if a1 == a2:
                sim_ad[a1][a2] = 1
            else:
                sim_ad[a1][a2],partial_user = get_item_sim(a1,a2,sim_user,partial_user,ad_user)
    i+=1


# In[5]:


def get_evidence(graph,items,evidence_type):
    evidence_item = {u:{} for u in items}
    for i1 in items:
        for i2 in items:
            if i1 == i2:
                evidence_item[i1][i2] = 1
            else:
                num_inter = len(list(set(graph[i1])&set(graph[i2])))
                if evidence_type == 'geo':
                    evidence_item[i1][i2] = sum([1/(2**i) for i in range(1, num_inter+1)])
                else:
                    evidence_item[i1][i2] = (1-math.exp(-num_inter))
    return evidence_item

# Evidence (a,b)
geo_evidence_user = get_evidence(user_ad,users,evidence_type='geo')
geo_evidence_ad = get_evidence(ad_user,ads,evidence_type='geo')
exp_evidence_user = get_evidence(user_ad,users,evidence_type='exp')
exp_evidence_ad = get_evidence(ad_user,ads,evidence_type='exp')


# In[6]:


#S_evidence
geo_sim_user = copy.deepcopy(sim_user)
geo_sim_ad = copy.deepcopy(sim_ad)
exp_sim_user = copy.deepcopy(sim_user)
exp_sim_ad = copy.deepcopy(sim_ad)

for u1 in users:
        for u2 in users:
            if u1 == u2:
                geo_sim_user[u1][u2] = 1
                exp_sim_user[u1][u2] = 1
            else:
                geo_sim_user[u1][u2] *= geo_evidence_user[u1][u2]
                exp_sim_user[u1][u2] *= exp_evidence_user[u1][u2]
for a1 in ads:
        for a2 in ads:
            if a1 == a2:
                geo_sim_ad[a1][a2] = 1
                exp_sim_ad[a1][a2] = 1
            else:
                geo_sim_ad[a1][a2] *= geo_evidence_ad[a1][a2]
                exp_sim_ad[a1][a2] *= exp_evidence_ad[a1][a2]


# In[27]:


# get similar items
def get_similar(sim_item,Q_item):
    #sorted_item = sorted(sim_item[Q_item].items(), key = lambda x:x[1],reverse=True)[1:]
    sim_rank = {}
    for key,sim in sim_item[Q_item].items():
        sim = round(sim,4)
        if sim not in sim_rank:
            sim_rank[sim] = [key]
        else:
            sim_rank[sim].append(key)
        
    return sorted(sim_rank.items(), reverse=True)[1:4]

def get_top_three(sim_rank):
    top = []
    for i in range(0,3):
        top.append(sorted(sim_rank[i][1], key = lambda x: int(x))[0])

    return top


# In[28]:


# get outputs of top 3
top_3_sim_user = [user[0] for score,user in get_similar(sim_user,Q_user)]
top_3_sim_ad = get_top_three(get_similar(sim_ad,Q_ad))
top_3_geo_user = [user[0] for score,user in get_similar(geo_sim_user,Q_user)]
top_3_geo_ad = get_top_three(get_similar(geo_sim_ad,Q_ad))
top_3_exp_user = [user[0] for score,user in get_similar(exp_sim_user,Q_user)]
top_3_exp_ad = get_top_three(get_similar(exp_sim_ad,Q_ad))
with open('output_b.txt','w') as f:
    f.write(','.join(top_3_sim_user)+'\n')
    f.write(','.join(top_3_sim_ad) + '\n')
    f.write(','.join(top_3_geo_user)+'\n')
    f.write(','.join(top_3_geo_ad) + '\n')
    f.write(','.join(top_3_exp_user)+'\n')
    f.write(','.join(top_3_exp_ad) )
    f.close()

