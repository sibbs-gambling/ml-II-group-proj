
import pandas as pd
import numpy as np

# Data provided by Eloy.
raw = pd.read_csv('../data/mergedCurrenciesClean.csv')
raw.describe()


# * 1 column with dates, 90 columns with -/USD exchange rates, 3 columns with precious metal prices.
# * 4833 rows, but many columns have a few NA values.
# * Some currencies e.g HKD have large missing ranges.
#
# * Denominated in USD, but we can use triangle rule to obtain the exchange rates between any pair:
#
# x/y = x/USD divided by y/USD

# ### Finding the time (row) range for the paper.

# In[3]:


# Identify paper start and end dates indices in data frame.
start = np.argmax(raw["YYYY/MM/DD"].values=="1/1/1999")
end = np.argmax(raw["YYYY/MM/DD"].values=="6/30/2008")
print(start)
print(end)


# * The paper begins with 1/1/1999, but the data frame starts from 1/4/1999. Not a big problem.
# * Row #2383 is the final date index.
#
# We will use the start and end indices to slice the data later.

# ### Storing the currencies for the paper in a set "paper_currencies".

# * The paper uses 60 currencies and gold, silver, and platinum.
# * The raw data has 90 currencies and gold, silver, and platinum.
#
# We need to find the 60 currencies used in the paper in the data frame.

# In[5]:


import re # Regular expressions (turn out to be useful!)

# text.txt is the pdf paper converted online into a text document.
fin = open("../docs/analysis_of_a_network_structure_of_FX.txt", 'rt', encoding='utf-8')
# Find all instances of 3 consecutive capital letters. e.g 'USD' 'AUD' 'XAU'
regex = r'[A-Z][A-Z][A-Z]'
matches = []
for line in fin:
    matches += re.findall(regex, line)
# Store matches.
# Note that this list contains many repetitions and ...
# ...some acronyms e.g 'ISO' (international standards organization)'MST' (minimal spanning tree)
paper_codes = matches
# print(paper_codes)
fin.close()

print("There are:", len(set(paper_codes)), "unique 3 letter codes in the paper")
print(set(paper_codes))


# Next, we extract currency codes from the data frame.

# In[6]:


# Store currency codes from raw data frame in set frame_codes.
colnames = list(raw.columns.values)[1:]
matches = []
for rate in colnames:
    matches += list(rate.split('/'))
frame_codes = matches

print("There are:", len(set(frame_codes)), "unique 3 letter codes in the data frame")
print(set(frame_codes))


# In[7]:


# Store codes that appear in both lists. We use sets to discard repetitions.
common_codes = set(paper_codes) & set(frame_codes)

print("There are:", len(set(common_codes)), "unique currency codes of interest")
print(set(common_codes))


# We have 62 of 63 assets. What are we missing?

# In[8]:


set(paper_codes) - set(common_codes) # Set difference.


# ISO and MST are acronyms and not currencies. However, **the data frame is missing VEB (Venezulan Bolivar). The ISO code changed to VEF in 2008.** Does the data frame include VEF?

# In[9]:


"VEF" in frame_codes


# In[10]:


# Store all currency codes of interest:
paper_currencies = common_codes.copy()
paper_currencies.add("VEF")
print(paper_currencies)
print("Number of final assets: ", len(paper_currencies))


# ### Identifing which columns (exchange rates) to extract.

# In[11]:


# Boolean indexing.
filter = [False for i in range(len(colnames))]
for i in range(len(colnames)):
    for code in common_codes:
        if re.search(code+r'/USD', colnames[i]) != None:
            filter[i] = True

print(filter)


# ### Extract data used by the paper.

# In[12]:


data = raw.iloc[start:end+1, filter]
data.describe()


# NOTE: The currencies have -/USD_x and -/USD_y exchange rates. These subscripts quantify the amount of commodity - nothing to worry about.

# ### Some tests in replicating the paper results.

# In[106]:


# Our test data drops some columns that are constant/ missing a significant amount of data.
# We then backfill to interpolate remaining NA values.
# We now have a clean (but perhaps not satisfactory) data frame to work with.
test = data.drop(columns=['ZMW/USD', 'PAB/USD', 'HRK/USD', 'EEK/USD', 'BSD/USD']).fillna(method='backfill')
test.to_csv('../data/cleaned_paper_data.csv')
test.describe()


# NOTE: We have only 58 columns remaining.

#  ## Computing a distance matrix using correlations
#
#  ### Q: Should we calculate log returns instead?
#
#  ### A: I am pretty sure we should use the log returns. In their paper they denote the log returns by $G_X^B$. When they talk about forming the correlation matrix they don't say they use that explicitly but it makes the most sense to me.

# In[105]:


C = test.corr()
D = np.sqrt(2*(1-C))  # isn't this supposed to be sqrt not square
# Print top left 5x5 submatrix.
print(D.iloc[:5, :5])


# ## Visualizing distances using 2D MDS
# (Like SML Homework 4.)

# In[95]:


import matplotlib.pyplot as plt

# Compute inner product matrix.
n = D.shape[0]
IminusM = np.eye(n)-np.ones((n, n))/n
B = -IminusM.dot(D**2).dot(IminusM)/2

# Transform featurespace using MDS.
from sklearn.decomposition import KernelPCA
mds = KernelPCA(kernel='precomputed', n_components=2)
Z = mds.fit_transform(B)

# Plot transformed feature space.
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(Z[:, 0], Z[:, 1], color="green")
ax.set_title("2D MDS of Scaling")
ax.set_xlabel("Z2")
ax.set_ylabel("Z1")

# Add text labels.
for i in range(n):
    plt.text(Z[i, 0], Z[i, 1], s=colnames[i])


# ## Minimal Spanning Trees
# Could generalize to spectral clustering.

# A minimal spanning tree is represented by a sparse distance matrix. Every pair of nodes is connected via a single path along edges. Therefore, with N nodes there are N-1 non-zero edges.
#
# **MSTs are used for their sparsity/interpretability**. Instead of (NC2) = N(N-1)/2 inter-node edges, we reduce the graph down to N-1. See paper for construction details. We may use other MDS techniques if needed.
#
# ~~**Currently, we have no way of visualizing the MST** as in the paper.~~
#
# We are using the $\texttt{networx}$ module to do this now.

# In[109]:


from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

MST = minimum_spanning_tree(D) # cst object. NOT a distance matrix.
MST_matrix = MST.toarray()
# This prints the non zero distances between indexed nodes (currencies deonominated in USD).

#print("Number of non-zero edges:", MST.shape[0])
#print(MST)

labels={}
for j in range(len(D.columns)):
    labels[j] = D.columns[j]
fig = plt.figure(figsize=(10, 10))
G = nx.Graph(MST_matrix)
pos = nx.fruchterman_reingold_layout(G)
nx.draw(G, with_labels=False, node_size=3000,
        node_color="skyblue", node_shape="o",
        alpha=0.5, pos=pos)
nx.draw_networkx_labels(G,pos,labels)
plt.show()


# ## Hierarchical Clustering Preliminaries
# ### Q: Transition to R?
# ### A: Probably a good idea if we decide to go the heirarchical clustering route.

# In[97]:


from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(D, 'complete') # Linkage matrix.

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, orientation='left', labels=colnames, leaf_font_size=20)
plt.show()
