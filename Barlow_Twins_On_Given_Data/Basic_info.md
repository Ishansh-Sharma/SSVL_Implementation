The given dataset was extremely large : 

130000 - images  
100 - classes   

Applying the methods of SSL was extremly hard on this dataset due to various reasons that have been documented in the "proposed_doc" file . Hence i will not discuss them again . 

The Barlow Twins approach for SSL has been implemented on a part of dataset 
Here i have used the first 40 classes to train and test 

nummber of images - 52000
number of classes - 40

The whole architecture has been kept very lightweight , due to computational limitations 
Hence , the results are not very good 

Other relevant information include : 
Time to train - 2.1 hours ( approx ) ( in the iteration with no memory error ) 
Used Kaggle GPU T4 x 2

relevant comments are present in the code .  

