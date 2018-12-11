---
title: "Census Population Segmentation"
output: html_document
---


-Problem Statement:
  
  How can clustering be applied to better understand the voters for each county?

-Project Objective
  
  The goal is to gain a better understanding of the voters in the United States. This project     explores how we can apply unsupervised clustering to better integrate science into the task of   identifying the voters which are more likely to vote for either Republican or Democrat.

-Purpose and Motivation

  The outcome of this analysis are natural groupings of similar counties in a transformed         feature space. The cluster that a county belongs to can be leveraged to plan an election        campaign, for example, to understand how to reach a group of similar counties by highlighting   messages that resonate with that group. More generally, this technique can be applied by        businesses in customer or user segmentation to create targeted marketing campaigns.

-Independent Variables
  
  -Total Population 
  
    1)Men
    2)Women 
    
  -Citizenship
  
  -Employment
  
    1)Private worker 
    2)Public worker
    3)Self Employed
    4)Unemployed
    
  -Race
  
    1)White
    2)Hispanic
    3)Black
    4)Asian
    5)Pacific
    6)Native

-Dependent Variables
  
  -County of the United States
  
-Graphs

![output_14_1](/Visuals/output_14_1.png)
![output_14_3](/Visuals/output_14_3.png)
![output_14_5](/Visuals/output_14_5.png)
![output_37_0](/Visuals/output_37_0.png)
![output_59_1](/Visuals/output_59_1.png)
![output_69_0](/Visuals/output_69_0.png)
  
-Methodolgy
  
  We divided the project into two phases; phase-I and phase-II. Phase-I consists of acquiring     and cleaning the dataset, whereas Phase-II reflects the modeling, visualization, scaling and    labeling of the data. 
  
-Phase - I
  
  -Acquiring and Cleaning Data
    In this step we got the data from        https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml. We extracted just enough to get meaningful insights from the dataset and keeping in mind the constraint of running the project on our computers.
    
  
  -Tidying Procedure
    “State” & “Census ID” were deleted, since state-county is the main focus. For which we           assigned “State-County” as the index.
    
-Phase - II

  
  -Modeling with Principal Component Analysis (PCA) Model
    
  We will be using principal component analysis (PCA) to reduce the dimensionality of our         data. This method decomposes the data matrix into features that are orthogonal with each        other. The resultant orthogonal features are linear combinations of the original feature        set. You can think of this method as taking many features and combining similar or redundant    features together to form a new, smaller feature set.
  
  -Visualizing the Data
  
   Here we analyzed our dataset with a mix of numerical and categorical columns. We can            visualize the data for some of our numerical columns and see what the distribution looks        like.
   
  -Data Scaling
  
   We need to standardize the scaling of the numerical columns in order to use any distance        based analytical methods so that we can compare the relative distances between different        feature columns. We can use minmaxscaler to transform the numerical columns so that they also    fall between 0 and 1.

  -Population Segmentation
    Now, we’ll use the Kmeans algorithm to segment the population of counties by the 5 PCA          attributes we have created. Kmeans is a clustering algorithm that identifies clusters of        similar counties based on their attributes. Since we have ~3000 counties and 34 attributes      in our original dataset, the large feature space may have made it difficult to cluster the      counties effectively. Instead, we have reduced the feature space to 5 PCA components, and       we’ll cluster on this transformed dataset.
  -K-Means Centroids
  -Heatmap of Centroids
  -Labeling the Clusters
  
-Conclusion
  
  By clustering a dataset using KMeans and after reducing the dimensionality using a PCA model,   we are able to improve the explainability of our modelling and draw an actionable conclusion.   Using these techniques, we have been able to better understand the essential characteristics    of different counties in the US and segment the electorate into groupings accordingly and       highlight Miami-Dade County's electorate characteristics.

-Refrences

https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml