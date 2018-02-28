from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.decomposition import PCA

gene_data  = pd.read_csv('ecs171.dataset.txt',delimiter='\t')
df = pd.DataFrame(gene_data)

#num_of_test_set = 39 # cannot be zero - CHOOSE HOW MUCH SAMPLE IN TEST SET

#test_sample = df.iloc[len(df)-num_of_test_set:len(df)]
#test_sample_growth = test_sample['GrowthRate']
#test_sample = test_sample.drop(['ID', 'Strain','Medium','Stress','GenePerturbed','GrowthRate'], axis=1)
# =============================================================================
# for i in range(0,num_of_test_set):
#     
#     test_sample.loc[i:,] = test_sample.iloc[i][6:,]
#     print('hi')
#     test_sample[i] =test_sample[i].values.reshape(1,-1)
# =============================================================================
#df = df.drop(df.index[len(df)-num_of_test_set:len(df)])


gene_exp_only = df.drop(['ID', 'Strain','Medium','Stress','GenePerturbed','GrowthRate'], axis=1)

growth = df['GrowthRate']
strain = df['Strain']
medium = df['Medium']
stress = df['Stress']
gene_perturbed = df['GenePerturbed']
strain_classes = list()
medium_classes = list()
stress_classes = list()
gene_perturbed_classes = list()

le = preprocessing.LabelEncoder()

le.fit(strain)
strain = le.transform(strain)


le.fit(medium)
medium = le.transform(medium)

le.fit(stress)
stress = le.transform(stress)

le.fit(gene_perturbed)
gene_perturbed = le.transform(gene_perturbed)




strain = pd.DataFrame(strain)
medium = pd.DataFrame(medium)
stress = pd.DataFrame(stress)
gene_perturbed = pd.DataFrame(gene_perturbed)


med_per = pd.concat([medium, stress], ignore_index=True)
le.fit(med_per)
med_per = le.transform(med_per)
med_per = pd.DataFrame(med_per)

strain_classes = strain.iloc[:,0].unique()
medium_classes = medium.iloc[:,0].unique()
stress_classes = stress.iloc[:,0].unique()
gene_perturbed_classes = gene_perturbed.iloc[:,0].unique()
med_per_classes = med_per.iloc[:,0].unique()



rsquared = list()
avg_r=  list()
x_iterator = list()
svm_data_index = list()

#Lasso model ========================================================================================
kf = KFold(n_splits=10, shuffle = True)#10 fold CV
for x in [x / 1000.0 for x in range(0, 60, 1)]:#steps from 0 to .1 with .01 steps <-------- change the / 1000.0 for more precise lambdas
    count = 0
    for train_index, test_index in kf.split(gene_exp_only):
        
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = gene_exp_only.iloc[train_index], gene_exp_only.iloc[test_index]
        y_train, y_test = growth.iloc[train_index], growth.iloc[test_index]
        
        rows, columns = X_train.shape
        rows2, columns2 = X_test.shape
        
        
        X_train = np.array(X_train).reshape(rows,columns,order = 'F')
        X_test = np.array(X_test).reshape(rows2,columns2,order = 'F')
        y_train = np.array(y_train).reshape(len(y_train),1,order = 'F')
        y_test = np.array(y_test).reshape(len(y_test),1,order = 'F')
        
        clf = Lasso(alpha=x, copy_X=True, fit_intercept=True, max_iter=1000, tol=0.001)
        clf.fit(X_train,y_train)
        result = clf.coef_
        
        
# =============================================================================
#         num_of_zero_coeff=0
#         for i in range(0,len(result)):
#             if(result[i]==0):
#                 num_of_zero_coeff +=1
#         
# =============================================================================
        predicted = clf.predict(X_test)
        result2 = mean_squared_error(y_test,predicted)
        rsquared.append(result2)
        count += 1
# =============================================================================
#         print('Number of features with non-zero coefficients in training set: ',(len(result)-num_of_zero_coeff)) #4435 for all data
#         print('Average error of the predicted growth values for the testing set #',count , ': ',sum(abs((y_test-predicted)))/len(predicted))
#         print('MSE #',count,': ',result2) 
#         
#         print('\n')
# =============================================================================
    
    avg_r.append(sum(rsquared)/len(rsquared))
    x_iterator.append(x)



avg_rr = (sum(avg_r)/len(avg_r))
temp_a = pd.DataFrame(avg_r)
temp_b = pd.DataFrame(x_iterator)
output = temp_a.join(temp_b,lsuffix='temp_a',rsuffix='temp_b')

print(output)
pyplot.plot(output.iloc[:,1],output.iloc[:,0])
pyplot.ylabel('Avg MSE of 10-fold CV')
pyplot.xlabel('lambda')
pyplot.show()

min_MSE = output.iloc[output.iloc[:,0].idxmin()][0]
opt_lambda = output.iloc[output.iloc[:,0].idxmin()][1]
print('Min MSE: ',min_MSE,'with lambda: ',opt_lambda)

clf2 = Lasso(alpha=opt_lambda, copy_X=True, fit_intercept=True, max_iter=1000, tol=0.001)
clf2.fit(gene_exp_only,growth)
opt_result = clf2.coef_


num_of_zero_coeff=0
for i in range(0,len(opt_result)):
    if(result[i]==0):
        num_of_zero_coeff +=1
    else:
        svm_data_index.append(i)
svm_data = gene_exp_only.iloc[:,svm_data_index]
svm_data = pd.DataFrame(svm_data) #non zero coeff genes
print('Number of features with non-zero coefficients in training set: ',(len(opt_result)-num_of_zero_coeff)) 

#Lasso_mean ===========================================================================================

mean_test = gene_exp_only.mean()

clf3 = Lasso(alpha=opt_lambda, copy_X=True, fit_intercept=True, max_iter=1000, tol=0.001)
clf3.fit(gene_exp_only,growth)

print('\nPredicted growth of mean sample: ',clf3.predict(mean_test.reshape(1,-1)))

#PCA ===========================================================================================
pca = PCA(n_components=3)
pca.fit(gene_exp_only)
reduc_feat = pca.transform(gene_exp_only)
reduc_feat = pd.DataFrame(reduc_feat)
#Linear SVM ==========================================================================================

kf = KFold(n_splits=10, shuffle = True)#10 fold CV
lb = preprocessing.LabelBinarizer()
strain_pred = list()

for class1 in strain_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(svm_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_1, X_test_1 = svm_data.iloc[train_index], svm_data.iloc[test_index]
        y_train_1, y_test_1 = strain.iloc[train_index], strain.iloc[test_index]
        SVC_strain = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_strain.fit(X_train_1, y_train_1)
        strain_pred0 = SVC_strain.decision_function(X_test_1)
        fpr, tpr, thresholds = roc_curve(y_test_1, strain_pred0[:, 1], pos_label=class1)
        precision, recall, _ = precision_recall_curve(y_test_1, strain_pred0[:, 1],pos_label=class1)
        
        SVC_strain_coeff = SVC_strain.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class1, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Strain')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class1, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Strain')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()
#PCA SVM STRAIN+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for class1 in strain_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(reduc_feat):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_1, X_test_1 = reduc_feat.iloc[train_index], reduc_feat.iloc[test_index]
        y_train_1, y_test_1 = strain.iloc[train_index], strain.iloc[test_index]
        SVC_strain = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_strain.fit(X_train_1, y_train_1)
        strain_pred0 = SVC_strain.decision_function(X_test_1)
        fpr, tpr, thresholds = roc_curve(y_test_1, strain_pred0[:, 1], pos_label=class1)
        precision, recall, _ = precision_recall_curve(y_test_1, strain_pred0[:, 1],pos_label=class1)
        
        SVC_strain_coeff = SVC_strain.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class1, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Strain -PCA-')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class1, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Strain -PCA-')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()

# =============================================================================
# 
#     
#     
# mean_strain_pred = np.divide(strain_pred,10)
# lb.fit(y_test_1)
# 
# y1 = lb.transform(y_test_1)
# n_classes1 = y1.shape[1]
# 
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes1):
#     fpr[i], tpr[i], _ = roc_curve(y1[:, i], strain_pred0[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# 
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(), strain_pred0.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# 
# =============================================================================
#-=-=--=--=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=--=-=--=-=-=--=-=-
for class2 in medium_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(svm_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_2, X_test_2 = svm_data.iloc[train_index], svm_data.iloc[test_index]
        y_train_2, y_test_2 = medium.iloc[train_index], medium.iloc[test_index]
        SVC_medium = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_medium.fit(X_train_2, y_train_2)
        medium_pred0 = SVC_medium.decision_function(X_test_2)
        fpr, tpr, thresholds = roc_curve(y_test_2, medium_pred0[:, 1], pos_label=class2)
        precision, recall, _ = precision_recall_curve(y_test_2, medium_pred0[:, 1],pos_label=class2)
        
        SVC_medium_coeff = SVC_medium.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class2, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Medium')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class2, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Medium')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()
#SVM MEDIUM PCA+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for class2 in medium_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(reduc_feat):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_2, X_test_2 = reduc_feat.iloc[train_index], reduc_feat.iloc[test_index]
        y_train_2, y_test_2 = medium.iloc[train_index], medium.iloc[test_index]
        SVC_medium = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_medium.fit(X_train_2, y_train_2)
        medium_pred0 = SVC_medium.decision_function(X_test_2)
        fpr, tpr, thresholds = roc_curve(y_test_2, medium_pred0[:, 1], pos_label=class2)
        precision, recall, _ = precision_recall_curve(y_test_2, medium_pred0[:, 1],pos_label=class2)
        
        SVC_medium_coeff = SVC_medium.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class2, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Medium -PCA-')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class2, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Medium -PCA-')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()
#-=-=--=--=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=--=-=--=-=-=--=-=-
for class3 in stress_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(svm_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_3, X_test_3 = svm_data.iloc[train_index], svm_data.iloc[test_index]
        y_train_3, y_test_3 = stress.iloc[train_index], stress.iloc[test_index]
        SVC_stress = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_stress.fit(X_train_3, y_train_3)
        stress_pred0 = SVC_stress.decision_function(X_test_3)
        fpr, tpr, thresholds = roc_curve(y_test_3, stress_pred0[:, 1], pos_label=class3)
        precision, recall, _ = precision_recall_curve(y_test_3, stress_pred0[:, 1],pos_label=class3)
        
        SVC_stress_coeff = SVC_stress.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class3, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Stress')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class3, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Stress')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()
#SVM STRESS PCA+++++++++++++++++++++++++++++++++++++++++++++________________________________________-------------------------
for class3 in stress_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(reduc_feat):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_3, X_test_3 = reduc_feat.iloc[train_index], reduc_feat.iloc[test_index]
        y_train_3, y_test_3 = stress.iloc[train_index], stress.iloc[test_index]
        SVC_stress = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_stress.fit(X_train_3, y_train_3)
        stress_pred0 = SVC_stress.decision_function(X_test_3)
        fpr, tpr, thresholds = roc_curve(y_test_3, stress_pred0[:, 1], pos_label=class3)
        precision, recall, _ = precision_recall_curve(y_test_3, stress_pred0[:, 1],pos_label=class3)
        
        SVC_stress_coeff = SVC_stress.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class3, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Stress -PCA-')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class3, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Stress -PCA-')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()


#-=-=--=--=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=--=-=--=-=-=--=-=-

for class4 in gene_perturbed_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(svm_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_4, X_test_4 = svm_data.iloc[train_index], svm_data.iloc[test_index]
        y_train_4, y_test_4 = gene_perturbed.iloc[train_index], gene_perturbed.iloc[test_index]
        SVC_gene_perturbed = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_gene_perturbed.fit(X_train_4, y_train_4)
        gene_perturbed_pred0 = SVC_gene_perturbed.decision_function(X_test_4)
        fpr, tpr, thresholds = roc_curve(y_test_4, gene_perturbed_pred0[:, 1], pos_label=class4)
        precision, recall, _ = precision_recall_curve(y_test_4, gene_perturbed_pred0[:, 1],pos_label=class4)
        
        SVC_gene_perturbed_coeff = SVC_gene_perturbed.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class4, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Gene Perturbed')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class4, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Gene Perturbed')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()

#SVM GENE PERT PCA---------------------------------------------------------------------------------------------------------------------

for class4 in gene_perturbed_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(reduc_feat):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_4, X_test_4 = reduc_feat.iloc[train_index], reduc_feat.iloc[test_index]
        y_train_4, y_test_4 = gene_perturbed.iloc[train_index], gene_perturbed.iloc[test_index]
        SVC_gene_perturbed = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_gene_perturbed.fit(X_train_4, y_train_4)
        gene_perturbed_pred0 = SVC_gene_perturbed.decision_function(X_test_4)
        fpr, tpr, thresholds = roc_curve(y_test_4, gene_perturbed_pred0[:, 1], pos_label=class4)
        precision, recall, _ = precision_recall_curve(y_test_4, gene_perturbed_pred0[:, 1],pos_label=class4)
        
        SVC_gene_perturbed_coeff = SVC_gene_perturbed.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class4, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Gene Perturbed -PCA-' )
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class4, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Gene Perturbed -PCA-')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()

#COMPISITE SVM =================================================================================================
for class5 in med_per_classes:
    tprs = []
    pres = []
    aucs = []
    aucs2 = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_fpr2 = np.linspace(0, 1, 100)
    i = 0
    for train_index, test_index in kf.split(svm_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_5, X_test_5 = svm_data.iloc[train_index], svm_data.iloc[test_index]
        y_train_5, y_test_5 = med_per.iloc[train_index], med_per.iloc[test_index]
        SVC_med_per = LinearSVC(dual=False, max_iter=1000, tol=0.0001)
        SVC_med_per.fit(X_train_5, y_train_5)
        med_per_pred0 = SVC_med_per.decision_function(X_test_5)
        fpr, tpr, thresholds = roc_curve(y_test_5, med_per_pred0[:, 1], pos_label=class5)
        precision, recall, _ = precision_recall_curve(y_test_5, med_per_pred0[:, 1],pos_label=class5)
        
        SVC_med_per_coeff = SVC_med_per.coef_
        tprs.append(interp(mean_fpr, fpr, tpr))
        pres.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        pres[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        aucs.append(roc_auc)
        aucs2.append(pr_auc)
        pyplot.figure(1)
        #pyplot.plot(fpr, tpr, lw=1, alpha=0.2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        pyplot.figure(2)
        pyplot.subplot(111)
        #pyplot.plot(recall, precision, lw=1, alpha=0.9, label='ROC fold %d (AUC = %0.2f)' % (i, pr_auc))
    
        i += 1
        
    pyplot.figure(1)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    

    
    pyplot.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class5, mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)    
    
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC and AUC for Medium and Stress')
    pyplot.legend(loc="lower right")
    #pyplot.show()
    
    pyplot.figure(2)   
    pyplot.subplot(111)
    
    mean_precision = np.mean(pres, axis=0)
    mean_precision[-1] = 1.0
    mean_auc2 = auc(mean_fpr2, mean_precision)
    std_auc2 = np.std(aucs2)
    
    pyplot.plot(mean_fpr2, mean_precision,
         label=r'Mean PR of class %0.2f (AUC = %0.2f $\pm$ %0.2f)' % (class5, mean_auc2, std_auc2),
         lw=2, alpha=.8)

    std_tpr2 = np.std(pres, axis=0)
    tprs_upper2 = np.minimum(mean_precision + std_tpr, 1)
    tprs_lower2 = np.maximum(mean_precision - std_tpr, 0)
    pyplot.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color='grey', alpha=.2)   
    
    pyplot.xlabel('Precision')
    pyplot.ylabel('Recall')
    pyplot.title('PR and AUC for Medium and Stress')
    pyplot.legend(loc="lower right")

pyplot.figure(1)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.figure(2)   
N = 2
params = pyplot.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*N, plSize[1]*N) )

pyplot.show()

#Ridge Model
# =============================================================================
# 
# kf = KFold(n_splits=10, shuffle = True)
# for train_index, test_index in kf.split(gene_exp_only):
#     
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = gene_exp_only.iloc[train_index], gene_exp_only.iloc[test_index]
#     y_train, y_test = growth.iloc[train_index], growth.iloc[test_index]
#     
#     rr = Ridge(alpha=20, copy_X=True, fit_intercept=True, max_iter=None, solver='auto', tol=0.001)
#     rr.fit(X_train,y_train)
#     result = rr.coef_
#     
#     
#     num_of_zero_coeff=0
#     for i in range(0,len(result)):
#         if(result[i]==0):
#             num_of_zero_coeff +=1
#     
#     predicted = rr.predict(X_test)
#     result2 = 1 - mean_squared_error(y_test,predicted)
#     rsquared.append(result2)
#     count += 1
#     print('Number of features with non-zero coefficients in training set: ',(len(result)-num_of_zero_coeff)) #4435 for all data
#     print('Average error of the predicted growth values for the testing set #',count , ': ',sum(abs((y_test-predicted)))/len(predicted))
#     print('1 - MSE #',count,': ',result2) 
#     
#     print('\n')
#     
# avg_r = sum(rsquared)/len(rsquared)
# print('Average 1-MSE: ',avg_r)
# =============================================================================
# load dataset

# configure bootstrap
n_iterations = 1000 #<--------- change number for more iterations
n1_size = int(len(gene_exp_only) * 0.50) #<--------- change .50 to whatever ratio of train to test for X
n2_size = int(len(growth) * 0.50)#<--------- change .50 to whatever ratio of train to test for Y
# run bootstrap
stats = list()
gene_exp_only = gene_exp_only.astype(float)
#cell = gene_exp_only.iloc[0].index
#print(gene_exp_only.iloc[gene_exp_only.index[0]])


#Bootstrapping Confidence Interval ================================================================================
for i in range(n_iterations):
	# prepare train and test sets
    seed=np.random.randint(0,1000000) 
    test = pd.DataFrame() 
    trainX = resample(gene_exp_only, n_samples=n1_size,random_state = seed)
    trainY = resample(growth, n_samples=n2_size,random_state = seed)
    testX = gene_exp_only[~gene_exp_only.index.isin(trainX.index)]
    testY = growth[~growth.index.isin(trainY.index)]
    #print(gene_exp_only.index == train.index)
# =============================================================================
#     for x in range(0, len(gene_exp_only)):
#         for y in range(0, len(train)):
#             if gene_exp_only.index[x] != train.index[y]:
#                 test = test.append(gene_exp_only.iloc[gene_exp_only.index[x]])
#     
# =============================================================================
    #for rows in range(0,len(gene_exp_only)):
        
        #if(gene_exp_only[rows].index not in train.index):
                #test = test.append(gene_exp_only[rows])
    

	# fit model
    
    model = Lasso(alpha=opt_lambda, copy_X=True, fit_intercept=True, max_iter=1000, tol=0.001)

    model.fit(trainX, trainY)
	# evaluate model
    predictions = model.predict(testX)
    testY = np.array(testY)
    score = mean_squared_error(testY,predictions)
    #print(score)
    stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval between %.1f%% and %.1f%% MSE' % (alpha*100, lower*100, upper*100))




#gene_exp_only = gene_exp_only.drop(df.index[len(df)-num_of_test_set:len(df)])
#growth = growth.drop(df.index[len(df)-num_of_test_set:len(df)])

