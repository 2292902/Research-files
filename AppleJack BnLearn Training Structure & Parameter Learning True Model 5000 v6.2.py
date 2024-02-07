#!/usr/bin/env python
# coding: utf-8

# ## AppleJack BnLearn Training Structure & Parameter Learning True Model 5000 v6.2

# In[90]:


import pgmpy as p
import bnlearn as bn
from pgmpy.models import BayesianNetwork
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from bnlearn import structure_learning
from bnlearn import parameter_learning
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta


# ### Training

# In[91]:


# Read the CSV file and skip the "Unnamed" column
ground_truth_df = pd.read_csv("GT168000_train_data.csv", index_col=0)
ground_truth_df.head()


# In[92]:


# Defining the model structure by passing a list of edges
model = BayesianModel([('LatVar1', 'Drought'),('LatVar2', 'Drought'),('Drought', 'DryFarm'), ('LatVar3', 'DryFarm'),('DryFarm', 'LowSoilMoisture'), ('LatVar4', 'LowSoilMoisture'),('LowSoilMoisture', 'DryTree'), ('DryTree', 'BadTreeCondition'), ('SickTree', 'BadTreeCondition'), ('BadTreeCondition', 'BadCropCondition'), ('LatVar5', 'BadCropCondition'), ('BadTreeCondition', 'LoseLeaves'), ('LatVar6', 'LoseLeaves')])
DAG =  bn.make_DAG(model)


# In[93]:


#Plot BN representing Directed Acycling Graph - DAG
bn.plot(DAG)


# In[94]:


# Defining all the conditional probabilities tables by a method in pgmpy called TabularCPD
cpd_ltv1 = TabularCPD(variable='LatVar1', variable_card=2, values=[[0.9000],[0.1000]])
cpd_ltv2 = TabularCPD(variable='LatVar2', variable_card=2, values=[[0.4000],[0.6000]])
cpd_ltv3 = TabularCPD(variable='LatVar3', variable_card=2, values=[[0.2000],[0.8000]])
cpd_ltv4 = TabularCPD(variable='LatVar4', variable_card=2, values=[[0.4500],[0.5500]])
cpd_ltv5 = TabularCPD(variable='LatVar5', variable_card=2, values=[[0.2000],[0.8000]])
cpd_ltv6 = TabularCPD(variable='LatVar6', variable_card=2, values=[[0.6000],[0.4000]])

## Representation of CPD in pgmpy, the columns are the evidence and rows are the states of the variable. 
#represents P(grade|diff, intel) 

cpd_dr = TabularCPD(variable='Drought', variable_card=2, 
                    values=[[0.9000, 0.6000, 0.4000, 0.1000],
                            [0.1000, 0.4000, 0.6000, 0.9000]],
                    evidence=['LatVar1','LatVar2'],
                    evidence_card=[2, 2])

cpd_df = TabularCPD(variable='DryFarm', variable_card=2, 
                    values=[[0.9000, 0.8000, 0.2000, 0.1000],
                            [0.1000, 0.2000, 0.8000, 0.9000]],
                    evidence=['Drought', 'LatVar3'],
                    evidence_card=[2, 2])

cpd_lsm = TabularCPD(variable='LowSoilMoisture', variable_card=2, 
                    values=[[0.6500, 0.2000, 0.8000, 0.3500],
                            [0.3500, 0.8000, 0.2000, 0.6500]],
                    evidence=['DryFarm', 'LatVar4'],
                    evidence_card=[2, 2])

cpd_dt = TabularCPD(variable='DryTree', variable_card=2, 
                    values=[[0.7500, 0.2500],
                            [0.2500, 0.7500]],
                    evidence=['LowSoilMoisture'],
                    evidence_card=[2])

cpd_st = TabularCPD(variable='SickTree', variable_card=2, values=[[0.7051], [0.2949]])

cpd_btc = TabularCPD(variable='BadTreeCondition', variable_card=2, 
                    values=[[0.7500, 0.6000, 0.4000, 0.2500],
                            [0.2500, 0.4000, 0.6000, 0.7500]],
                    evidence=['DryTree','SickTree'],
                    evidence_card=[2, 2])

cpd_bcc = TabularCPD(variable='BadCropCondition', variable_card=2, 
                    values=[[0.7000, 0.6000, 0.4000, 0.3000],
                            [0.3000, 0.4000, 0.6000, 0.7000]],
                    evidence=['BadTreeCondition', 'LatVar5'],
                    evidence_card=[2, 2])

cpd_ll = TabularCPD(variable='LoseLeaves', variable_card=2, 
                    values=[[0.7500, 0.3500, 0.6500, 0.2500 ],
                            [0.2500, 0.6500, 0.3500, 0.7500]],
                    evidence=['BadTreeCondition', 'LatVar6'],
                    evidence_card=[2, 2])


# In[95]:


# Add the conditional probability tables to the model
model.add_cpds(cpd_ltv1, cpd_ltv2, cpd_ltv3, cpd_ltv4, cpd_ltv5, cpd_ltv6, cpd_dr, cpd_df, cpd_lsm, cpd_dt, cpd_st, cpd_btc, cpd_bcc, cpd_ll)


# In[96]:


# Check if the model is valid
model.check_model()


# In[97]:


# Create a BayesianModelSampling object with the model
sampler = BayesianModelSampling(model)


# In[49]:


# Use Bayesian Parameter Estimation to refine the parameter estimates
# Set the number of samples and Dirichlet hyperparameters
n = 5000


# In[98]:


# Generate 5000 samples from the model using forward sampling
samples = sampler.forward_sample(size=5000)


# In[99]:


# Convert the samples to DataFrame format
samples_df = pd.DataFrame(samples)


# In[100]:


df = samples_df


# In[101]:


# Write the DataFrame to a CSV file
df.to_csv("GTGN5000_train_data.csv", index=False)


# In[102]:


# Read the CSV file and skip the "Unnamed" column
ground_truth_df = pd.read_csv("GTGN5000_train_data.csv", index_col=0)
ground_truth_df.head()


# In[103]:


# Defining the model structure by passing a list of edges
model = BayesianModel([('LatVar1', 'Drought'),('LatVar2', 'Drought'),('Drought', 'DryFarm'), ('LatVar3', 'DryFarm'),('DryFarm', 'LowSoilMoisture'), ('LatVar4', 'LowSoilMoisture'),('LowSoilMoisture', 'DryTree'), ('DryTree', 'BadTreeCondition'), ('SickTree', 'BadTreeCondition'), ('BadTreeCondition', 'BadCropCondition'), ('LatVar5', 'BadCropCondition'), ('BadTreeCondition', 'LoseLeaves'), ('LatVar6', 'LoseLeaves')])
DAG =  bn.make_DAG(model)


# In[104]:


model = bn.structure_learning.fit(ground_truth_df)
G = bn.plot(model)


# ###  BIC Scores

# In[105]:


# Define the HillClimbSearch object
hc = HillClimbSearch(ground_truth_df)


# In[106]:


# Perform the structure learning with BIC scoring
scoring_method = BicScore(ground_truth_df)
best_model = hc.estimate(scoring_method)


# In[107]:


# Print the learned structure (edges)
print("Learned edges:")
print(best_model.edges())


# In[108]:


# Create a Directed Graph (DiGraph) to visualize the Bayesian network
G = nx.DiGraph()

# Add nodes to the graph
G.add_nodes_from(best_model.nodes())

# Add edges to the graph based on the learned structure
G.add_edges_from(best_model.edges())

# Plot the Bayesian network
pos = nx.spring_layout(G, seed=42)  # You can choose different layout algorithms
nx.draw(G, pos, with_labels=True, node_size=200, node_color="Skyblue", font_size=10, font_color="black", font_weight="regular")
plt.title("Learned Bayesian Network")
plt.show()


# In[109]:


# Perform the structure learning with BIC scoring
scoring_method = BicScore(ground_truth_df)
bic_scores = []


# In[110]:


# Create BicScore object
bic_score = BicScore(ground_truth_df)


# In[111]:


# Calculate and print BIC score for the given network structure
bic_network_score = bic_score.score(best_model)


# In[112]:


# Number of iterations
num_iterations = 10

for _ in range(num_iterations):
    # Estimate the best model
    best_model = hc.estimate(scoring_method)
    
    # Calculate BIC score for the given network structure
    bic_network_score = scoring_method.score(best_model)
    bic_scores.append(bic_network_score)


# In[113]:


# Calculate mean and standard deviation of BIC scores
bic_mean = np.mean(bic_scores)
bic_std_dev = np.std(bic_scores)


# In[114]:


print(f"Mean BIC Score: {bic_mean}")
print(f"Standard Deviation of BIC Scores: {bic_std_dev}")


# In[115]:


model_update = bn.parameter_learning.fit(model, ground_truth_df, methodtype='bayes')


# ### Validation KL Divergence 

# ### BadTreeCondition & BadCropCondition

# In[116]:


from scipy.special import kl_div


# In[117]:


# Generate 5000 samples from the ground truth model
ground_truth_samples = ground_truth_df.sample(n=5000, random_state=1)
ground_truth_samples_df = pd.DataFrame(ground_truth_samples)


# In[118]:


# Create empty lists to store KL Divergence values
kl_divergence_list1 = []
kl_divergence_list2 = []


# In[119]:


# Number of iterations
num_iterations = 10


# In[120]:


# Number of iterations
num_iterations = 10
sample_size = 5000  # Sample size should not exceed the population size

for _ in range(num_iterations):
    # Take n samples from the CPD distribution
    data_df = bn.sampling(model_update, n=sample_size, methodtype='bayes')
    
    # Generate samples from the ground truth model
    ground_truth_samples = ground_truth_df.sample(n=sample_size, random_state=1)
    ground_truth_samples_df = pd.DataFrame(ground_truth_samples)
    
    # Perform KL divergence analysis for the full conditional distribution between the baseline model and the ground truth model
    kl_divergence1 = kl_div(ground_truth_samples_df['BadTreeCondition'].value_counts(normalize=True),
                           data_df['BadTreeCondition'].value_counts(normalize=True))
    
    kl_divergence_list1.append(kl_divergence1)
    
    # Perform KL divergence analysis for another distribution (you can add more as needed)
    kl_divergence2 = kl_div(ground_truth_samples_df['BadCropCondition'].value_counts(normalize=True),
                           data_df['BadCropCondition'].value_counts(normalize=True))
    
    kl_divergence_list2.append(kl_divergence2)


# In[121]:


# Calculate mean and standard deviation of KL Divergence values
kl_divergence_mean1 = np.mean(kl_divergence_list1)
kl_divergence_std_dev1 = np.std(kl_divergence_list1)

kl_divergence_mean2 = np.mean(kl_divergence_list2)
kl_divergence_std_dev2 = np.std(kl_divergence_list2)


# In[122]:


print(f"Mean KL Divergence 1: {kl_divergence_mean1}")
print(f"Standard Deviation KL Divergence 1: {kl_divergence_std_dev1}")

print(f"Mean KL Divergence 2: {kl_divergence_mean2}")
print(f"Standard Deviation KL Divergence 2: {kl_divergence_std_dev2}")


# ### BIC Score Validation Data

# In[123]:


# Read the CSV file and skip the "Unnamed" column
ground_truth_df = pd.read_csv("GT72000_val_data.csv", index_col=0)
ground_truth_df.head()


# In[125]:


# Defining the model structure by passing a list of edges
model = BayesianModel([('Drought', 'LatVar2'), ('DryFarm', 'Drought'), ('DryFarm', 'LatVar3'), ('DryFarm', 'LatVar4'), ('LatVar3', 'Drought'), ('LowSoilMoisture', 'LatVar4'), ('LowSoilMoisture', 'DryFarm'), ('DryTree', 'LowSoilMoisture'), ('DryTree', 'SickTree'), ('BadTreeCondition', 'DryTree'), ('BadTreeCondition', 'BadCropCondition'), ('BadTreeCondition', 'SickTree'), ('BadCropCondition', 'LatVar5'), ('LoseLeaves', 'LatVar6'), ('LoseLeaves', 'BadTreeCondition')])
DAG =  bn.make_DAG(model)


# In[126]:


# Use Bayesian Parameter Estimation to refine the parameter estimates
# Set the number of samples and Dirichlet hyperparameters
n = 5000


# In[127]:


# Generate 5000samples from the model using forward sampling
samples = sampler.forward_sample(size = 5000)


# In[128]:


# Convert the samples to DataFrame format
samples_df = pd.DataFrame(samples)


# In[129]:


print(samples_df)


# In[131]:


# Write the DataFrame to a CSV file
df.to_csv("GTGN5000_validation_data.csv", index=False)


# In[132]:


# Read the CSV file and skip the "Unnamed" column
ground_truth_df = pd.read_csv("GTGN5000_validation_data.csv", index_col=0)
ground_truth_df.head()


# In[136]:


model = bn.structure_learning.fit(ground_truth_df)
G = bn.plot(model)


# In[134]:


# Print the learned structure (edges)
print("Learned edges:")
print(best_model.edges())


# In[135]:


# Create a Directed Graph (DiGraph) to visualize the Bayesian network
G = nx.DiGraph()

# Add nodes to the graph
G.add_nodes_from(best_model.nodes())

# Add edges to the graph based on the learned structure
G.add_edges_from(best_model.edges())

# Plot the Bayesian network
pos = nx.spring_layout(G, seed=42)  # You can choose different layout algorithms
nx.draw(G, pos, with_labels=True, node_size=200, node_color="Skyblue", font_size=10, font_color="black", font_weight="regular")
plt.title("Learned Bayesian Network")
plt.show()


# ###  BIC Scores

# In[46]:


# Define the HillClimbSearch object
hc = HillClimbSearch(data_df)


# In[47]:


# Perform the structure learning with BIC scoring
scoring_method = BicScore(data_df)
bic_scores = []


# In[48]:


# Perform the structure learning with BIC scoring
scoring_method = BicScore(data_df)
best_model = hc.estimate(scoring_method)


# In[49]:


# Number of iterations
num_iterations = 10

for _ in range(num_iterations):
    # Estimate the best model
    best_model = hc.estimate(scoring_method)
    
    # Calculate BIC score for the given network structure
    bic_network_score = scoring_method.score(best_model)
    bic_scores.append(bic_network_score)


# In[50]:


print(f"Mean BIC Score: {bic_mean}")
print(f"Standard Deviation of BIC Scores: {bic_std_dev}")


# In[51]:


model_update = bn.parameter_learning.fit(model, ground_truth_df, methodtype='bayes')


# ### Validation KL Divergence 1000

# ### BadTreeCondition & BadCropCondition

# In[52]:


from scipy.special import kl_div


# In[53]:


# KL Divergence
# Generate 1000 samples from the ground truth model
ground_truth_samples = data_df.sample(n=1000, random_state=1)
ground_truth_samples_df = pd.DataFrame(ground_truth_samples)


# In[54]:


# Create empty lists to store KL Divergence values
kl_divergence_list1 = []
kl_divergence_list2 = []


# In[55]:


# Number of iterations
num_iterations = 10


# In[56]:


# Number of iterations
num_iterations = 10
sample_size = 1000  # Sample size should not exceed the population size

for _ in range(num_iterations):
    # Take n samples from the CPD distribution
    data_df = bn.sampling(model_update, n=sample_size, methodtype='bayes')
    
    # Generate samples from the ground truth model
    ground_truth_samples = ground_truth_df.sample(n=sample_size, random_state=1)
    ground_truth_samples_df = pd.DataFrame(ground_truth_samples)
    
    # Perform KL divergence analysis for the full conditional distribution between the baseline model and the ground truth model
    kl_divergence1 = kl_div(ground_truth_samples_df['BadTreeCondition'].value_counts(normalize=True),
                           data_df['BadTreeCondition'].value_counts(normalize=True))
    
    kl_divergence_list1.append(kl_divergence1)
    
    # Perform KL divergence analysis for another distribution (you can add more as needed)
    kl_divergence2 = kl_div(ground_truth_samples_df['BadCropCondition'].value_counts(normalize=True),
                           data_df['BadCropCondition'].value_counts(normalize=True))
    
    kl_divergence_list2.append(kl_divergence2)


# In[57]:


# Calculate mean and standard deviation of KL Divergence values
kl_divergence_mean1 = np.mean(kl_divergence_list1)
kl_divergence_std_dev1 = np.std(kl_divergence_list1)

kl_divergence_mean2 = np.mean(kl_divergence_list2)
kl_divergence_std_dev2 = np.std(kl_divergence_list2)


# In[58]:


print(f"Mean KL Divergence 1: {kl_divergence_mean1}")
print(f"Standard Deviation KL Divergence 1: {kl_divergence_std_dev1}")

print(f"Mean KL Divergence 2: {kl_divergence_mean2}")
print(f"Standard Deviation KL Divergence 2: {kl_divergence_std_dev2}")


# In[ ]:





# In[ ]:




