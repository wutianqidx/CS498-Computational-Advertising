#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


# get node types
sports, politics, arts = list(range(30)), list(range(30,70)), list(range(70,100))

# Read a graph from a list of edges
network_1 = nx.read_edgelist('network1_edges.txt', nodetype=int, create_using=nx.DiGraph())
network_2 = nx.read_edgelist('network2_edges.txt', nodetype=int, create_using=nx.DiGraph())

# Return the PageRank of the nodes in the graph
page_rank_1 = nx.pagerank(network_1, alpha=1, max_iter=100)
page_rank_2 = nx.pagerank(network_2, alpha=1, max_iter=100)


# In[3]:


##Qustion1
sports_rank = []
sorted_rank_1 = sorted(page_rank_1.items(), key=lambda kv: kv[1], reverse=True )
for node in sorted_rank_1:
    if node[0] in sports:
        sports_rank.append(node[0])
top_10_sports = sports_rank[:10]


# In[4]:


print(top_10_sports)


# In[5]:


##Qustion2
politics_rank = []
sorted_rank_2 = sorted(page_rank_2.items(), key=lambda kv: kv[1], reverse=True )
for node in sorted_rank_2:
    if node[0] in politics:
        politics_rank.append(node[0])
top_10_politics = politics_rank[:10]


# In[6]:


print(top_10_politics)


# In[7]:


##Qustion3
variation_1 = []
nstart = None
for i in range(100):
    pagerank_1 = nx.pagerank(network_1, alpha=1, max_iter=1, tol=1, nstart=nstart)
    variation_1.append(pagerank_1[1])
    nstart = pagerank_1

plt.title('Variation of node 1 in network1')
plt.xlabel('iterations')
plt.ylabel('pagerank')

plt.plot(variation_1)


# In[8]:


##Qustion4
variation_2 = []
nstart = None
for i in range(100):
    pagerank_2 = nx.pagerank(network_2, alpha=1, max_iter=1, tol=1, nstart=nstart)
    variation_2.append(pagerank_2[1])
    nstart = pagerank_2

plt.title('Variation of node 1 in network2')
plt.xlabel('iterations')
plt.ylabel('pagerank')

plt.plot(variation_2)


# In[ ]:




