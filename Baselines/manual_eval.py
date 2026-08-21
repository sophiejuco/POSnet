#import packages
import pandas as pd


#read data - save data in same directory as .py file
data = pd.read_csv('test_true_tags_man.csv', index_col=0)


#reduce data to subsection that was manually tagged
data_short = data.iloc[0:1693,:].copy()


#separate manual tags and true tags
man_tags = data_short.iloc[:,1].copy()
true_tags = data_short.iloc[:,2].copy()


#function to generate metrics
#expects lists of the same length
def tag_metrics(actual_arr, predicted_arr):
    
    len_actual = len(actual_arr)-(actual_arr.isna().sum()) #don't count None values
    len_pred = len(predicted_arr)-(predicted_arr.isna().sum()) #don't count None values
    num_correct = 0
    
    for i in range(len(actual_arr)):
        if predicted_arr[i] == actual_arr[i]:
            num_correct += 1
        else:
            continue
            
    #precision
    precision = num_correct/len_pred
    
    #recall
    recall = num_correct/len_actual
    
    #f-measure
    f1 = 2/((1/precision)+(1/recall))
    
    print("precision: " + str(round(precision, 4)))
    print("recall: " + str(round(recall, 4)))
    print("f1: " + str(round(f1, 4)))


#run evaluation
tag_metrics(true_tags, man_tags)


# In[ ]:




