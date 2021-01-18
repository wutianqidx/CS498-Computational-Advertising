#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine


# In[ ]:


## Read Input
user_item_rating = {}
item_user_rating = {}
item_content = {}
total_rating = []
vocab = []
test = []
with open('input.txt') as f:
    R,M = f.readline().strip().split(' ')
    for i,lines in enumerate(f):
        line = lines.strip().split(' ')
        if  i < int(R):
            user,item,rating = int(line[0]),int(line[1]),float(line[2])
            total_rating.append(float(rating))
            if user not in user_item_rating:
                user_item_rating[user] = {item:rating}
            else:
                user_item_rating[user][item] = rating
            if item not in item_user_rating:
                item_user_rating[item] = {user:rating}
            else:
                item_user_rating[item][user] = rating

        if  i >= int(R) and i < int(R) + int(M):
            item,content = int(line[0]),line[1:]
            item_content[item] = content
            
        if  i >= int(R) + int(M):
            test.append((line[0],line[1]))
    f.close()


# In[ ]:


## Mean
mu = sum(total_rating)/len(total_rating)


# In[ ]:


## bias
b_i = {item:0 for item in item_user_rating}
b_u = {user:0 for user in user_item_rating}

for i in item_user_rating.keys():
    rating = []
    for u in item_user_rating[i]:
        rating.append(item_user_rating[i][u] - mu)
    b_i[i] = sum(rating)/len(rating)
    
for u in user_item_rating.keys():
    rating = []
    for i in user_item_rating[u]:
        rating.append(user_item_rating[u][i] - mu - b_i[i])
    b_u[u] = sum(rating)/len(rating)


# In[ ]:


## similarity_function
def Pearson_correlation(i,j):
    U_ij = set(item_user_rating[i].keys())&set(item_user_rating[j].keys())
    if len(U_ij) == 0:
        return 0
    top = 0
    bot = 0
    for u in U_ij:
        b_ui = b_u[u]+b_i[i]+mu
        b_uj = b_u[u]+b_i[j]+mu
        top += (user_item_rating[u][i]-b_ui)*(user_item_rating[u][j]-b_uj)
        bot += (user_item_rating[u][i]-b_ui)**2*(user_item_rating[u][j]-b_uj)**2
    s_ij = top/(bot)**(1/2)
    return(s_ij)


# In[ ]:


total_content = ' '.join([' '.join(list(set(content))) for content in item_content.values()])
df = Counter(total_content.split())
vocab = list(df.keys())
N = len(item_content)
idf = {x: np.log(N/(df[x])+1) for x in df}


# In[ ]:


def tf_idf(i):
    tf_top_i = Counter(item_content[i])
    d_i = np.zeros(len(vocab))
    for word in item_content[i]:
        d_i[vocab.index(word)] = tf_top_i[word]/len(item_content[i]) * idf[word]
    return(d_i)

def Content_similarity(i,j):
    d_i,d_j = tf_idf(i),tf_idf(j)
    s_ij = 1 - cosine(d_i,d_j)
    return(s_ij)
    


# In[ ]:


## predict
def predict_score(u,i,sim_func=Pearson_correlation):
    top = 0
    bot = 0
    for j in user_item_rating[u].keys():
        b_uj = b_u[u]+b_i[j]+mu
        s_ij = sim_func(i,j)
        top += s_ij*(user_item_rating[u][j]-b_uj)
        bot += s_ij
    score = b_u[u] + b_i[i] + mu + top/bot
    return round(score,1)      


# In[ ]:


## output
print('Pearson correlation')
print('user\t', 'item\t', 'score\t')
for user,item in test:
    print(user,'\t',item,'\t',predict_score(int(user),int(item)))


# In[ ]:


print('Content similarity')
print('user\t', 'item\t', 'score\t')
for user,item in test:
    print(user,'\t',item,'\t',predict_score(int(user),int(item),sim_func=Content_similarity))


# In[ ]:




