import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations, product
import json
import sys
from itertools import product
from scipy import stats


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

def checkRule(col,y,rule):
    col=np.array(col)
    mean=np.mean(col)
    std=np.std(col)
    if rule=="lowerOut":
        values=np.argwhere(col<mean-std)
    if rule=="higherOut":
        values=np.argwhere(col>mean+std)
    if rule=="lowerIn":
        values=np.argwhere((col<mean) & (col>mean-std))
    if rule=="higherIn":
        values=np.argwhere((col>mean) & (col<mean+std))
    return [values.flatten() ,mean, std]


def main():
    try:
        dataset=sys.argv[1]
    except:
        dataset="segment"
        #dataset="pima_diabetes"
        #dataset="iris"
    #df=pd.read_csv('Dataset/train/pima_diabetes.csv', sep=',',header=None)
    df=pd.read_csv('../Dataset/train/'+dataset+'.csv', sep=',',header=None)
    y=np.array(df.iloc[:,-1])
    df=df.iloc[:,:-1]
    df=convertData(df)
    print(df)

    colNames=list(df.columns)
    output={}
    output["stats"]={}
    output["rules"]={}
    columnRules={}
    rules=["lowerOut","higherOut","lowerIn","higherIn"]
    for i in colNames:
        for r in rules:
            try:
                x=checkRule(df[i],y,r)
                columnRules[i][r]=x[0]
                output["stats"]["col"+str(i)]["mean"]=x[1]
                output["stats"]["col"+str(i)]["std"]=x[2]
            except:
                columnRules[i]={}
                x=checkRule(df[i],y,r)
                columnRules[i][r]=x[0]
                output["stats"]["col"+str(i)]={}
                output["stats"]["col"+str(i)]["mean"]=x[1]
                output["stats"]["col"+str(i)]["std"]=x[2]

    rows=list(range(len(df)))
    count=0
    for i in range(1,len(colNames)+1):
        rulesC=list(product(rules,repeat=i))
        namesC=list(combinations(colNames,i))
        per=0
        for rule in rulesC:
            if per%50==0:
                print("through columns:"+str((i/(len(colNames)+1))*100))
                print("through rules:"+str((per/len(rulesC))*100))
                print(1-(len(rows)/len(df)))
                print("\n\n")
            per=per+1
            for names in namesC:
                tempRows=rows
                #print(rule)
                #print(names)
                cCombo=""
                for x in range(len(rule)): #goes through each rule in combo
                    name=int(names[x])
                    setting=rule[x]
                    yRules=columnRules[name][setting]
                    tempRows=np.intersect1d(yRules,tempRows)
                    if len(tempRows)==0:
                        break
                    cCombo=cCombo+str(name)+","+setting+","
                if len(tempRows)!=0:
                    if len(set(y[tempRows]))==1:
                        #print(cCombo)
                        #print(y[tempRows][0])
                        output["rules"][count]={}
                        output["rules"][count]["output"]=y[tempRows][0]
                        output["rules"][count]["percentage"]=len(tempRows)/len(df)
                        output["rules"][count]["LabelPercentage"]=list(y[tempRows]).count(y[tempRows][0])/list(y).count(y[tempRows][0])
                        output["rules"][count]["rows"]=tempRows.tolist()
                        output["rules"][count]["name"]=cCombo[:-1]
                        output["rules"][count]["forced"]="no"
                        count=count+1
                        rows=[i for i in rows if i not in tempRows]
                    else: 
                        if i==len(colNames):
                            if len(tempRows)!=0:
                                tempOutput=stats.mode(y[tempRows])[0][0]
                                addRows=[]
                                for lastRow in tempRows:
                                    if y[lastRow]==tempOutput:
                                        addRows.append(lastRow)
                                rows=[i for i in rows if i not in addRows]
                                output["rules"][count]={}
                                output["rules"][count]["output"]=y[addRows][0]
                                output["rules"][count]["percentage"]=len(addRows)/len(df)
                                output["rules"][count]["LabelPercentage"]=list(y[tempRows]).count(y[addRows][0])/list(y).count(y[addRows][0])
                                output["rules"][count]["rows"]= [int(z) for z in addRows]
                                output["rules"][count]["name"]=cCombo[:-1]
                                output["rules"][count]["forced"]="yes"
                                count=count+1

            if len(rows)==0:
                break


    print(rows)
    print(len(rows)/len(df))
    output["trainingTotalPercentage"]=(1-len(rows)/len(df))*100
    output["trainingMissedLines"]=rows
    


    with open('training_'+dataset+".txt", 'w') as convert_file:
         convert_file.write(json.dumps(output))
    print (output)

if __name__ == "__main__":
    main()