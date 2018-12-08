
# coding: utf-8


import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from matplotlib import pyplot
from sklearn.multiclass import OneVsRestClassifier

print("FORMATTING DATA") 

### FORMATTING DATA ###
data=pd.read_csv('diabetic_data.csv')
diag=data[['diag_1','diag_2','diag_3']]
data=data.drop(['admission_type_id','discharge_disposition_id','admission_source_id','max_glu_serum','number_inpatient','number_emergency','number_outpatient','encounter_id','patient_nbr','weight','payer_code','A1Cresult','medical_specialty','max_glu_serum','diag_1','diag_2','diag_3'],axis=1)
data=data.drop(data[data.race=='?'].index)
data=data.drop(data[data.gender=='Unknown/Invalid'].index)
read=data['readmitted']
read=read.str.upper()
read=read.replace(['NO','<30','>30'],[0,1,1])
data=data.replace(['None','Female','No','Down'],0)
data=data.replace(['Male','Yes','Up','Steady'],1)
data=data.replace(['Down'],2)
r=pd.get_dummies(data[['race','diabetesMed']])
change=pd.get_dummies(data['change'])
a=pd.get_dummies(data['age'])
newdata=pd.concat([data,r,a,change],axis=1)
newdata=newdata.drop(['age','race','readmitted','change','diabetesMed'],axis=1)
X=data.drop(['readmitted'],axis=1)
read=np.array(read)
########################

#### SPLITTING DATA ####
x, X_test, y, y_test = train_test_split(newdata, read, test_size = 0.15, random_state = 0)
X_train,x_dev,y_train,y_dev=train_test_split(x,y,test_size=.18, random_state=0)
########################


print("Base Classifier") 


#### Base rules ####

train_X, test_X, train_y, test_y = train_test_split(data, read, test_size=0.3,random_state=0)
#print(train_X)
trainX=np.array(test_X)
trainy=np.array(test_y)

#print(trainX)
answers=[]
for thing in trainX:
   # print(len(thing))
    #num lab, diabetes meds, time in hospital <4
    if thing[2]==1 and thing[3]>=4 and thing[4]>=40 and thing[31]=='Yes':
       # print(thing[31])
        answers.append(1)
    else:
        answers.append(0)
i=0
#trainX=list(trainX)
test=0
for answer in answers:
    if answer==trainy[i]:
        test+=1
    i+=1
print("accuracy:",test/len(answers))


print("Logistic Regression") 


#### Logistic Regression ####
models = []
#grid search along solver and C values
sol=['newton-cg', 'lbfgs', 'liblinear', 'sag']
cs=[.0001,.001,.01,.1,1,10,20,50,100,1000]
for s in sol:
    print("Testing",s)
    for c in cs:
        model = LogisticRegression(C=c,solver=s)
        model.fit(X_train,y_train)
        pred=(model.predict(x_dev))
        testy=np.array(y_dev)
        f = f1_score(testy,pred)
        acc=accuracy_score(pred,testy)
        print("  C =",c,": Accuracy:",round(acc,5),"F1:",round(f,5))
        models.append((model,acc,f))
        
top_acc = max(models,key=itemgetter(1))[1]
top_acc_f1 = max(models,key=itemgetter(1))[2]
top_model_acc = max(models,key=itemgetter(1))[0]

print('\nBest model score by accuracy:',top_acc,"\nfrom model:  ",top_model_acc)
print('\nCorresponding f1:',top_acc_f1)


print("RANDOM FOREST") 
#### RANDOM FOREST ####


#hyperparameter grid search 
mxDepth = [5,10,15] #max_depth 
n_ests = [10,15,20] #n_estimators
best_models = []
for n in n_ests:
    print("n_estimators =",n)
    maximized_models=[]
    for mD in mxDepth:
        results = []
        print("\tmax_depth =",mD)
        for _ in range(51):
            model = RandomForestClassifier(n_estimators=n,max_depth=mD)
            model.fit(X_train,y_train)
            pred = model.predict(x_dev)
            testy=np.array(y_dev)
            print('\t'+str(_)+" iterations",end="\r",flush=True)

            results.append((model,accuracy_score(pred,testy)))
        print('\n\tBest model score:',max(results,key=itemgetter(1))[1])
        maximized_models.append((max(results,key=itemgetter(1))[0],max(results,key=itemgetter(1))[1],mD))

    print(n,"estimators","\tbest at max_depth =",max(maximized_models,key=itemgetter(1))[2],"\nscore =",max(maximized_models,key=itemgetter(1))[1])
    print("\n")
    best_models.append((max(maximized_models,key=itemgetter(1))[0],max(maximized_models,key=itemgetter(1))[1],n))

#grid search derived best model
final_best = max(best_models,key=itemgetter(1))[0]
f1 = f1_score(y_dev, final_best.predict(x_dev))

print("Best accuracy:",max(best_models,key=itemgetter(1))[1]," f1:",f1)
plt.bar(range(1,len(final_best.feature_importances_)+1),final_best.feature_importances_,ls="None")
plt.xlabel("Variable")
plt.ylabel("Importance (%)")
plt.title("Variable Importances")
plt.show()
for idx, column in enumerate(newdata.columns):
    print(idx+1, column, end="   ")


print("LinearSVC") 
#### LinearSVC reliability diagram with calibration ####

# fit a model
model = LinearSVC()
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)
# predict probabilities
probs = calibrated.predict_proba(x_dev)[:, 1]
# reliability diagram
fop, mpv = calibration_curve(y_dev, probs, n_bins=10, normalize=True)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
pyplot.plot(mpv, fop, marker='.')
pyplot.show()


#Predict and find accuracy
pred=(calibrated.predict(x_dev))
print("f1:",f1_score(pred,y_dev))
print("accuracy:",accuracy_score(pred,y_dev))
print("model:",calibrated)


print("LinearSVC with BaggingClassifier") 
#### SVC BaggingClassifier with  calibration ####

n_estimators = 10
# fit a model
model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

#Predict and find accuracy
pred=(calibrated.predict(x_dev))
print("f1:",f1_score(pred,y_dev))
print("accuracy:",accuracy_score(pred,y_dev))
print("model:",calibrated)


print("Best Prediction") 
#Random Forest yeilded the most accurate model
#Now we will run our model on the test set to get a final score
pred_final = final_best.predict(X_test)
print("\nfinal accuracy score:",accuracy_score(pred_final,np.array(y_test))," f1: ",f1_score(y_test,pred_final))
print("model:",final_best)

