#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Dependencies

# In[30]:


import os
import boto3
import io
import sagemaker

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.style.use('ggplot')
import pickle, gzip, urllib, json
import csv


# In[31]:


from sagemaker import get_execution_role
role = get_execution_role()


# In[32]:


role


# ## Loading the dataset
# I have previously downloaded and stored the data in a public S3 bucket that you can access. You can usethe Python SDK to interact with AWS using a Boto3 client.

# In[33]:


raw_data_filename = 'census.csv'
s3 = boto3.resource('s3')
bucket='censuspop'
s3.Bucket(bucket).download_file(raw_data_filename, 'census.csv')

counties = pd.read_csv('census.csv')
pd.set_option('display.max_rows', 20)


# In[34]:


counties.head()


# ## Step 2: Exploratory data analysis *EDA* - Data cleaning and exploration
# ### a. Cleaning the data
# We can do simple data cleaning and processing right in our notebook instance, using the compute instance of the notebook to execute these computations.
# 
# How much data are we working with?
# 
# There are 3220 rows with 37 columns

# In[35]:


counties.shape


# Let's just drop any incomplete data to make our analysis easier. We can see that we lost 2 rows of incomplete data, we now have 3218 rows in our data.

# In[36]:


counties.dropna(inplace=True)
counties.shape


# Let's combine some of the descriptive reference columns such as state and county and leave the numerical feature columns.
# 
# We can now set the 'state-county' as the index and the rest of the numerical features become the attributes of each unique county.

# In[37]:


counties.index=counties['State'] + "-" + counties['County']
counties.head()
drop=["CensusId" , "State" , "County"]
counties.drop(drop, axis=1, inplace=True)
counties.head()


# ### b. Visualizing the data
# Now we have a dataset with a mix of numerical and categorical columns. We can visualize the data for some of our numerical columns and see what the distribution looks like.
# 
# For example, from the figures above you can observe the distribution of counties that have a percentage of workers in Professional, Service, or Office occupations. Viewing the histograms can visually indicate characteristics of these features such as the mean or skew. The distribution of Professional workers for example reveals that the typical county has around 25-30% Professional workers, with a right skew, long tail and a Professional worker % topping out at almost 80% in some counties.

# In[38]:


import seaborn as sns

for a in ['Professional', 'Service', 'Office']:
    ax=plt.subplots(figsize=(6,3))
    ax=sns.distplot(counties[a])
    title="Histogram of " + a
    ax.set_title(title, fontsize=12)
    plt.show()


# For example, from the figures above you can observe the distribution of counties that have a percentage of workers in Professional, Service, or Office occupations. Viewing the histograms can visually indicate characteristics of these features such as the mean or skew. The distribution of Professional workers for example reveals that the typical county has around 25-30% Professional workers, with a right skew, long tail and a Professional worker % topping out at almost 80% in some counties.

# ### c. Feature engineering
# 
# **Data Scaling**- We need to standardize the scaling of the numerical columns in order to use any distance based analytical methods so that we can compare the relative distances between different feature columns. We can use minmaxscaler to transform the numerical columns so that they also fall between 0 and 1.

# In[39]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
counties_scaled=pd.DataFrame(scaler.fit_transform(counties))
counties_scaled.columns=counties.columns
counties_scaled.index=counties.index


# We can see that all of our numerical columns now have a min of 0 and a max of 1.

# In[40]:


counties_scaled.describe()


# ## Step 3: Data modelling
# ### a. Dimensionality reduction
# 
# For data modeling this model uses principal component analysis (PCA) to reduce the dimensionality of our data. This method decomposed the data matrix into features that are orthogonal with each other. The resultant orthogonal features were linear combinations of the original feature set. You can think of this method as taking many features and combining similar or redundant features together to form a new, smaller feature set.
# 
# Firstly it called the PCA model then specified different parameters of the model. These can be resource configuration parameters, such as how many instances to use during training, or what type of instances to use or model computation hyperparameters, such as how many components to use when performing PCA. 

# In[41]:


from sagemaker import PCA
bucket_name='pcacounties'
num_components=33

pca_SM = PCA(role=role,
          train_instance_count=1,
          train_instance_type='ml.c4.xlarge',
          output_path='s3://'+ bucket_name +'/counties/',
            num_components=num_components)


# Next, we prepare data for Amazon SageMaker by extracting the numpy array from the DataFrame and explicitly casting to  `float32`

# In[42]:


train_data = counties_scaled.values.astype('float32')


# The *record_set* function in the Amazon SageMaker PCA model converts a numpy array into a record set format that is the required format for the input data to be trained. This is a requirement for all Amazon SageMaker built-in models. The use of this data type is one of the reasons that allows training of models within Amazon SageMaker to perform faster, for larger data sets compared with other implementations of the same models, such as the sklearn implementation.
# 
# We call the fit function on our PCA model, passing in our training data, and this spins up a training instance or cluster to perform the training job.

# In[43]:


get_ipython().run_cell_magic('time', '', 'pca_SM.fit(pca_SM.record_set(train_data))')


# ### b. Accessing the PCA model attributes
# 
# After the model is created, we can also access the underlying model parameters.
# 
# Once that was done the ND array was loaded using MXNet.

# In[46]:


job_name='pca-2018-12-02-23-50-54-143'
model_key = "counties/" + job_name + "/output/model.tar.gz"

boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# After the model is unzipped and decompressed, we can load the ND array using MXNet.

# In[47]:


import mxnet as mx
pca_model_params = mx.ndarray.load('model_algo-1')


# **Three groups of model parameters are contained within the PCA model.**
# 
# **mean**: is optional and is only available if the “subtract_mean” hyperparameter is true when calling the training step from the original PCA function.
# 
# **v**: contains the principal components (same as ‘components_’ in the sklearn PCA model).
# 
# **s**: the singular values of the components for the PCA transformation. This does not exactly give the % variance from the original feature space, but can give the % variance from the projected feature space.
#              
#    explained-variance-ratio ~= square(s) / sum(square(s))
#  
# To calculate the exact explained-variance-ratio vector if needed, it simply requires saving the sum of squares of the original data (call that N) and computing explained-variance-ratio = square(s) / N.

# In[48]:


s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())


# We now calculated the variance explained by the largest n components that we wanted to keep. For this example, let's take the top 5 components.
# 
# We can see that the largest 5 components explain ~72% of the total variance in our dataset:

# In[49]:


s.iloc[28:,:].apply(lambda x: x*x).sum()/s.apply(lambda x: x*x).sum()


# After that we decided to keep the top 5 components, in order to take only the 5 largest components from our original s and v matrix.

# In[50]:


s_5=s.iloc[28:,:]
v_5=v.iloc[:,28:]
v_5.columns=[0,1,2,3,4]


# We can now examine the makeup of each PCA component based on the weightings of the original features that are included in the component. For example, the following code shows the first component. We can see that this component describes an attribute of a county that has high poverty and unemployment, low income and income per capita, and high Hispanic/Black population and low White population.
# 
# Note that this is v_5[4] or last component of the list of components in v_5, but is actually the largest component because the components are ordered from smallest to largest. So v_5[0] would be the smallest component. Similarly, change the value of component_num to cycle through the makeup of each component.

# In[51]:


component_num=1

first_comp = v_5[5-component_num]
comps = pd.DataFrame(list(zip(first_comp, counties_scaled.columns)), columns=['weights', 'features'])
comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
ax=sns.barplot(data=comps.sort_values('abs_weights', ascending=False).head(10), x="weights", y="features", palette="Blues_d")
ax.set_title("PCA Component Makeup: #" + str(component_num))
plt.show()


# Similarly, you can go through and examine the makeup of each PCA components and try to understand what the key positive and negative attributes are for each component. The following code names the components, but feel free to change them as you gain insight into the unique makeup of each component.

# In[52]:


#PCA_list=['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5']

PCA_list=["Poverty/Unemployment", "Self Employment/Public Workers", "High Income/Professional & Office Workers", "Black/Native Am Populations & Public/Professional Workers", "Construction & Commuters"]


# ### c. Deploying the PCA model
# 
# We can now deploy this model endpoint and use it to make predictions. 
# 

# In[28]:


get_ipython().run_cell_magic('time', '', "pca_predictor = pca_SM.deploy(initial_instance_count=1, \n                                 instance_type='ml.t2.medium')")


# We can also pass our original dataset to the model so that we can transform the data using the model we created. Then we can take the largest 5 components and this will reduce the dimensionality of our data from 34 to 5.

# In[29]:


get_ipython().run_cell_magic('time', '', 'result = pca_predictor.predict(train_data)')


# In[30]:


counties_transformed=pd.DataFrame()
for a in result:
    b=a.label['projection'].float32_tensor.values
    counties_transformed=counties_transformed.append([list(b)])
counties_transformed.index=counties_scaled.index
counties_transformed=counties_transformed.iloc[:,28:]
counties_transformed.columns=PCA_list


# Now we have created a dataset where each county is described by the 5 principle components that we analyzed earlier. Each of these 5 components is a linear combination of the original feature space. We can interpret each of these 5 components by analyzing the makeup of the component shown previously.

# In[31]:


counties_transformed.head()


# ### d. Population segmentation using unsupervised clustering
# 
# Now, we’ll use the Kmeans algorithm to segment the population of counties by the 5 PCA attributes we have created. Kmeans is a clustering algorithm that identifies clusters of similar counties based on their attributes. Since we have ~3000 counties and 34 attributes in our original dataset, the large feature space may have made it difficult to cluster the counties effectively. Instead, we have reduced the feature space to 5 PCA components, and we’ll cluster on this transformed dataset.

# In[32]:


train_data = counties_transformed.values.astype('float32')


# First, we call and define the hyperparameters of our KMeans model as we have done with our PCA model. The Kmeans algorithm allows the user to specify how many clusters to identify. In this instance, let's try to find the top 7 clusters from our dataset.

# In[33]:


from sagemaker import KMeans

num_clusters = 7
kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type='ml.c4.xlarge',
                output_path='s3://'+ bucket_name +'/counties/',              
                k=num_clusters)


# Then we train the model on our training data.

# In[34]:


get_ipython().run_cell_magic('time', '', 'kmeans.fit(kmeans.record_set(train_data))')


# Now we deploy the model and we can pass in the original training set to get the labels for each entry. This will give us which cluster each county belongs to.

# In[35]:


get_ipython().run_cell_magic('time', '', "kmeans_predictor = kmeans.deploy(initial_instance_count=1, \n                                 instance_type='ml.t2.medium')")


# In[36]:


get_ipython().run_cell_magic('time', '', 'result=kmeans_predictor.predict(train_data)')


# We can see the breakdown of cluster counts and the distribution of clusters.

# In[37]:


cluster_labels = [r.label['closest_cluster'].float32_tensor.values[0] for r in result]


# In[38]:


pd.DataFrame(cluster_labels)[0].value_counts()


# In[39]:


ax=plt.subplots(figsize=(6,3))
ax=sns.distplot(cluster_labels, kde=False)
title="Histogram of Cluster Counts"
ax.set_title(title, fontsize=12)
plt.show()


# However, to improve explainability, we need to access the underlying model to get the cluster centers. These centers will help describe which features characterize each cluster.

# ## Step 4: Drawing conclusions from our modelling
# 
# In this step we combined the PCA and Kmeans along with the information contained in the model to form a conclusion. 

# ### a. Accessing the KMeans model attributes
# 
# First, we will go into the bucket where the kmeans model is stored and extract it.

# In[40]:


job_name='kmeans-2018-12-03-00-08-15-552'
model_key = "counties/" + job_name + "/output/model.tar.gz"

boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')


# In[44]:


Kmeans_model_params = mx.ndarray.load('model_algo-1')


# **There is 1 set of model parameters that is contained within the KMeans model.**
# 
# **Cluster Centroid Locations**: The location of the centers of each cluster identified by the Kmeans algorithm. The cluster location is given in our PCA transformed space with 5 components, since we passed the transformed PCA data into the model.

# In[45]:


cluster_centroids=pd.DataFrame(Kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns


# In[46]:


cluster_centroids


# We plotted a heatmap of the centroids and their location in the transformed feature space which gave us insight into what characteristics define each cluster. Often with unsupervised learning, results are hard to interpret. This is one way to make use of the results of PCA plus clustering techniques together. Since we were able to examine the makeup of each PCA component, we were able to understand what each centroid represents in terms of the PCA components that we intepreted previously.
# 
# For example, we can see that cluster 1 has the highest value in the "Construction & Commuters" attribute while it has the lowest value in the "Self Employment/Public Workers" attribute compared with other clusters. Similarly, cluster 4 has high values in "Construction & Commuters," "High Income/Professional & Office Workers," and "Self Employment/Public Workers."

# In[53]:


plt.figure(figsize = (16, 6))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()


# We also mapped the cluster labels back to each individual county and examine which counties were naturally grouped together.

# In[48]:


counties_transformed['labels']=list(map(int, cluster_labels))
counties_transformed.head()


# Now, we can examine one of the clusters in more detail, like cluster 1 for example. A cursory glance at the location of the centroid tells us that it has the highest value for the "Construction & Commuters" attribute. We can now see which counties fit that description.

# In[61]:


cluster=counties_transformed[counties_transformed['labels']==5]
cluster


# ## Conclusion
# 
# By clustering a dataset using KMeans and after reducing the dimensionality using a PCA model, we are able to improve the explainability of our modelling and draw an actionable conclusion. Using these techniques, we have been able to better understand the essential characteristics of different counties in the US and segment the electorate into groupings accordingly and highlight Miami-Dade County's electorate characteristics.

# In[ ]:


sagemaker.Session().delete_endpoint(pca_predictor.endpoint)


# In[ ]:


sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)

