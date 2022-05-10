from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
import pandas as pd
import numpy as np
import sys

def convertData(df):
    label_encoder = LabelEncoder()
    count=0
    temp=df
    for i in df:
        count=count+1
        if type(df[i][0])!=str: #normalize columns
            if min(np.array(df.iloc[:, i]))<0:
                column = np.array(df.iloc[:, i])+abs(np.array(df.iloc[:, i]))
                temp.iloc[:,i]=column    
            column = np.array(df.iloc[:, i])/max(df.iloc[:, i])
            temp.iloc[:,i]=column
        else: #one hot encode strings, not used with current data sets but developed in case would be necessary
            #if count==len(df.columns)-1: #dont run on last column since it is ussually class
            column=df.iloc[:,i]
            integer_encoded = label_encoder.fit_transform(column)
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            temp=temp.drop(i, axis=1)
            temp=pd.concat([temp, pd.DataFrame(onehot_encoded)], axis=1)
    columnNames=list(range(len(temp.columns)))
    temp.columns = columnNames

    return temp

def checkRule(col,y,rule, mean,std):
    col=np.array(col)
    #mean=np.mean(col)
    #std=np.std(col)
    if rule=="above":
        values=np.argwhere(col>mean)
    if rule=="less":
        values=np.argwhere(col<mean)
    if rule=="lowerOut":
        values=np.argwhere(col<mean-std)
    if rule=="higherOut":
        values=np.argwhere(col>mean+std)
    if rule=="lowerIn":
        values=np.argwhere((col<mean) & (col>mean-std))
    if rule=="higherIn":
        values=np.argwhere((col>mean) & (col<mean+std))
    return values


try:
    dataset=sys.argv[1]
except:
    #dataset="iris"
    #dataset="pima_diabetes"
    dataset="Fire"
df=pd.read_csv('../Dataset/test/'+dataset+'.csv', sep=',',header=None)
#df=pd.read_csv('Dataset/test/'+dataset+'.csv', sep=',',header=None)
print(df)
y=np.array(df.iloc[:,-1])
df=df.iloc[:,:-1]

df=convertData(df)
print(df)



with open('training_'+dataset+'.txt', 'r') as fp:
    data = json.load(fp)

colNames=list(df.columns)
output={}
output["stats"]={}
output["rules"]={}
columnRules={}
rules=["above","less","lowerOut","higherOut","lowerIn","higherIn"]
classification={}
for i in colNames:
    for r in rules:
        try:
            x=checkRule(df[i],y,r,data["stats"]["col"+str(i)]["mean"],data["stats"]["col"+str(i)]["std"])
            columnRules[i][r]=x
        except:
            columnRules[i]={}
            x=checkRule(df[i],y,r,data["stats"]["col"+str(i)]["mean"],data["stats"]["col"+str(i)]["std"])
            columnRules[i][r]=x
            output["stats"]["col"+str(i)]={}

rows=list(range(len(df)))
for i in data["rules"]:
    rulesset=data["rules"][i]["name"].split(",")
    tempRows=rows
    for j in range(int(len(rulesset)/2)):
        yRules=columnRules[int(rulesset[j*2])][rulesset[j*2+1]]
        tempRows=np.intersect1d(yRules,tempRows)
    rows=[i for i in rows if i not in tempRows]
    label=data["rules"][i]["output"]
    if(len(tempRows)/len(df)*100>1):
        print(rulesset)
        print("coverage of dataset:")
        print(len(tempRows)/len(df)*100)
        print("coverage of label:")
        print(list(y[tempRows]).count(label)/list(y).count(label)*100)
        print("precision:")
        if len(tempRows)!=0:
            print(list(y[tempRows]).count(label)/len(tempRows))
        else:
            print("no instances")    
        print("\n\n")
    try:
        classification[label]=np.append(classification[label],tempRows)
    except Exception as e: 
        classification[label]=tempRows
totalTP=0
for i in classification:
    label=i
    corrLabels=y[classification[i]]
    print(label)
    #print(corrLabels)
    print("precision:")
    labelTP=list(corrLabels).count(label)
    labelFP=len(corrLabels)-list(corrLabels).count(label)
    labelFN=list(y).count(label)-labelTP
    labelTN=len(y)-(labelTP+labelFN+labelFN)
    print(labelTP/(labelTP+labelFP))
    precision=labelTP/(labelTP+labelFP)
    print("recall:")
    recall=labelTP/(labelTP+labelFN)
    print(labelTP/(labelTP+labelFN))
    totalTP=labelTP+totalTP
    print("F1:")
    f1=2*(precision*recall)/(precision+recall)
    print(f1)


print("covered:")
print (1-len(rows)/len(df))
print ("overall accuaracy:")
print (totalTP/len(y))
print ("total accuaracy of covered:")
print (totalTP/(len(y)-len(rows)))        
print("missed lines:")
print(rows)
