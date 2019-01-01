"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('disease_diagnosis.csv',
                      names=['patient_no','blood_type','gender','family_history','doing_sports',
                                                   'smoke','age','having_disease',])#Import all columns omitting the fist which consists the names of the animals
#We drop the patient_no since this is not a good feature to split the data on
dataset=dataset.drop('patient_no',axis=1)
###########################################################################################################
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    print ('Count in target data set', elements , counts)
    temp1 = []
    for i in range(len(elements)):
        val1 = (-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))
        temp1 = temp1 + [val1]
    entropy = np.sum(temp1)
    return entropy
###########################################################################################################

###########################################################################################################
def InfoGain(data, split_attribute_name, target_name="class"):
    print (split_attribute_name)
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    print ('Total Ent:', total_entropy)

    #Calculate the values and the corresponding counts for the split attribute
    vals, counts= np.unique(data[split_attribute_name],return_counts=True)
    print (split_attribute_name, ', Corresponding vals: ',vals, ' Count of vals:',counts)
    #Calculate the weighted entropy
    temp1 = []
    for i in range(len(vals)):
        p_s_a = (counts[i]/np.sum(counts))
        ent_p_s_a = entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name])
        val1 = p_s_a*ent_p_s_a
        print('attribute=', split_attribute_name,' A=', vals[i], ', p(S|A)=', p_s_a, ', Ent(S|A)=', ent_p_s_a)
        temp1 += [val1]
    Weighted_Entropy = np.sum(temp1)
    print ('Weighted Ent of ', split_attribute_name,'is', Weighted_Entropy)
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    print('Info Gain of S, A:', Information_Gain)
    return Information_Gain

###########################################################################################################
###########################################################################################################
def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.

    elif len(features) ==0:
        return parent_node_class

    #If none of the above holds true, grow the tree!

    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        #print(features)
        #Select the feature which best splits the dataset
        #print (data)
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        #print (item_values)
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        #print(best_feature)
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        return tree


        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        #Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)

            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return(tree)


"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###########################################################################################################
###########################################################################################################
def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)#We drop the index respectively relabel the index
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data
training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

"""
Train the tree, Print the tree and predict the accuracy
"""
tree = ID3(training_data,training_data,training_data.columns[:-1],'having_disease')