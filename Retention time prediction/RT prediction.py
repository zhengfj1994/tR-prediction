"""
The retention time prediction tool is referenced from internet. Thanks to the original author!
    Website：https://blog.csdn.net/sunyaowu315/article/details/82982989
    Original author: CS正阳
Adapted by Fujian Zheng <zhengfj@dicp.ac.cn>
Function: Prediction of metabolites retention time using in-house database
Description: Random forest classification is used to determine the dead time peak,
             and then random forest regression is used to predict the retention time.
             Parameter optimization using Bayesian optimization.
Input：In_houseDB_RT_MDs_2.csv and HMDB_MDs.csv
"""

"""
1. Import module
"""
import numpy as np,pandas as pd,os,seaborn as sns,matplotlib.pyplot as plt
import re
import random
import warnings
warnings.filterwarnings("ignore")

"""
2. Define functions
"""
# Reading csv file.
# In-house database is data_train and SMILES is located at the first column, RT is located at the last column.
# Public database is data_test and SMILES is located at the first column.
def readdb(filepath,dbname):
    os.chdir(filepath)
    dbdata = pd.read_csv(dbname)
    dbdata = dbdata.replace('',' ')
    return dbdata

def missvalueprecent(data):
    '''
    #==============================================================================
    # 分析数据中的缺失值情况。
    #==============================================================================
    '''
    miss_data = data.isnull().sum().sort_values(ascending=False)  # 缺失值数量
    total = data.isnull().count()  # 总数量
    miss_data_tmp = (miss_data / total).sort_values(ascending=False)  # 缺失值占比
    # 定义一个添加百分号的函数
    def precent(X):
        X = '%.2f%%' % (X * 100)
        return X
    miss_precent = miss_data_tmp.map(precent)
    # 小数转百分数
    miss_precent = miss_data_tmp.map(precent)
    # 根据缺失值占比倒序排序
    miss_data_precent = pd.concat([total, miss_precent, miss_data_tmp], axis=1, keys=[
        'total', 'Percent', 'Percent_tmp']).sort_values(by='Percent_tmp', ascending=False)
    # 有缺失值的变量打印出来
    print(miss_data_precent[miss_data_precent['Percent'] != '0.00%'])
    return miss_data_precent

def missvalueprocess(data,precent):
    '''
    # 将缺失值比例大于precent*100%的数据全部删除，剩余数值型变量用众数填充、类别型变量用None填充。
    '''
    drop_columns = miss_data_precent[miss_data_precent['Percent_tmp'] > precent].index
    data = data.drop(drop_columns, axis=1)
    # 类别型变量
    class_variable = [col for col in data.columns if data[col].dtypes == 'O']
    # 数值型变量
    numerical_variable = [col for col in data.columns if data[col].dtypes != 'O']  # 大写o
    print('类别型变量:%s' % class_variable, '数值型变量:%s' % numerical_variable)
    # 数值型变量用中位数填充，test集中最后一列为预测保留时间，所以不可以填充
    # Imputer填充模块
    from sklearn.preprocessing import Imputer
    # 选择填充方法为中位数填充
    padding = Imputer(strategy='median')
    data[numerical_variable] = padding.fit_transform(data[numerical_variable])
    data[class_variable] = data[class_variable].fillna('None') # 类别变量用None填充
    return data, numerical_variable

def variablefilter(data,numerical_variable,relativescore):
    '''
    #==============================================================================
    # 因为变量较多，直接采用关系矩阵，查看各个变量和因变量之间的关系，使用的时候采用spearman系数，原因:
    # Pearson 线性相关系数只是许多可能中的一种情况，为了使用Pearson 线性相关系数必须假设数
    # 据是成对地从正态分布中取得的，并且数据至少在逻辑范畴内必须是等间距的数据。如果这两条件
    # 不符合，一种可能就是采用Spearman 秩相关系数来代替Pearson 线性相关系数。Spearman 秩相关系
    # 数是一个非参数性质（与分布无关）的秩统计参数，由Spearman 在1904 年提出，用来度量两个变
    # 量之间联系的强弱(Lehmann and D’Abrera 1998)。Spearman 秩相关系数可以用于R 检验，同样可以
    # 在数据的分布使得Pearson 线性相关系数不能用来描述或是用来描述或导致错误的结论时，作为变
    # 量之间单调联系强弱的度量。
    # Spearman对原始变量的分布不作要求，属于非参数统计方法，适用范围要广些。
    # 理论上不论两个变量的总体分布形态、样本容量的大小如何，都可以用斯皮尔曼等级相关来进行研究 。
    #==============================================================================
    '''
    # ==============================================================================
    # 变量处理 ：
    #
    # 在变量处理期间，我们先考虑处理更简单的数值型变量，再考虑处理复杂的类别型变量；
    #
    # 其中数值型变量，需要先考虑和因变量的相关性，其次考虑变量两两之间的相关性，再考虑变量的多重共线性；
    #
    # 类别型变量除了考虑相关性之外，需要进行编码。
    # ==============================================================================
    # 计算变量之间的相关性
    numerical_variable_corr = data[numerical_variable].corr('spearman')
    numerical_corr = numerical_variable_corr[numerical_variable_corr['RT'] > relativescore]['RT']
    print(numerical_corr.sort_values(ascending=False))
    index0 = numerical_corr.sort_values(ascending=False).index
    # 结合考虑两两变量之间的相关性
    print(data_train[index0].corr('spearman'))
    # ==============================================================================
    # 结合上述情况，选择出相关性大于precent的变量：
    # ==============================================================================
    new_numerical = index0.drop('RT', 1)
    return new_numerical, index0

def dataprepareforclassifier(data_train,index,deadtime):
    import copy
    MDs_RT_OSI = data_train[index]  # OSI数据库物质和保留时间相关性大于0.5的分子识别符和保留时间
    MDs_RT_OSI['SMILES'] = data_train['SMILES']  # OSI数据库物质和保留时间相关性大于0.5的分子识别符和保留时间和SMILES
    RT_OSI = data_train['RT']  # OSI数据库的保留时间
    RT_classifier = copy.deepcopy(RT_OSI)  # 用于RandomForestClassifier训练的保留时间
    RT_classifier[RT_classifier <= deadtime] = 0  # 保留时间小于某值的认为为死时间出峰，设置为0
    RT_classifier[RT_classifier > deadtime] = 1  # 保留时间大于某值的认为为非死时间出峰，设置为1
    # 将OSI(分子识别符，保留时间，SMILES)和已经替换为0/1的保留时间随机分为训练集和验证集
    MDs_RT_OSI = MDs_RT_OSI.drop(['RT','SMILES'], 1, inplace=False)  # 去除RT和SMILES
    return MDs_RT_OSI, RT_classifier

def deadtimeclassifier(x, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    def rfc_cv(n_estimators, min_samples_split, max_features, max_depth):
        rfc = RandomForestClassifier(n_estimators=int(n_estimators),
                                     min_samples_split=int(min_samples_split),
                                     max_features=min(max_features, 0.999),  # float
                                     max_depth=int(max_depth),
                                     random_state=0,  ### 参数由2改为0
                                     n_jobs=-1)
        scores = cross_val_score(rfc, x, y, cv=5, scoring="accuracy")
        score = scores.mean()
        return score
    from bayes_opt import BayesianOptimization
    rfc_bo = BayesianOptimization(
        rfc_cv,
        {
            'n_estimators': (10, 1000),
            'min_samples_split': (2, 100),
            'max_features': (0.1, 0.999),
            'max_depth': (5, 300)
        }
    )
    rfc_bo.maximize()

    best_max_depth = rfc_bo.res[0]['params']['max_depth']
    best_max_features = rfc_bo.res[0]['params']['max_features']
    best_min_samples_split = rfc_bo.res[0]['params']['min_samples_split']
    best_n_estimators = rfc_bo.res[0]['params']['n_estimators']
    highest_score = rfc_bo.res[0]['target']

    for i in rfc_bo.res:  # 获取RandomForestClassifier最佳参数
        if i['target'] > highest_score:
            highest_score = i['target']
            best_max_depth = i['params']['max_depth']
            best_max_features = i['params']['max_features']
            best_min_samples_split = i['params']['min_samples_split']
            best_n_estimators = i['params']['n_estimators']

    # 通过贝叶斯优化找到最佳的参数后，代入模型，模型完成
    rfc = RandomForestClassifier(max_depth=best_max_depth, max_features=best_max_features,
                                 min_samples_split=best_min_samples_split.astype(int),
                                 n_estimators=best_n_estimators.astype(int))
    rfc.fit(x, y)
    return rfc

def dataprepareforregressor(data_train,index,deadtime):
    MDs_RT_OSI = data_train[index]
    x_regressor = MDs_RT_OSI[MDs_RT_OSI['RT'] > deadtime]
    y_regressor = x_regressor['RT']
    x_regressor = x_regressor.drop('RT', 1, inplace=False)  # 去除RT和SMILES
    return x_regressor, y_regressor

def trregressor(x,y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    def rfr_cv(n_estimators, min_samples_split, max_features, max_depth):
        rfr = RandomForestRegressor(n_estimators=int(n_estimators),
                                    min_samples_split=int(min_samples_split),
                                    max_features=min(max_features, 0.999),  # float
                                    max_depth=int(max_depth),
                                    random_state=0,  ### 参数由2改为0
                                    n_jobs=-1)
        scores = cross_val_score(rfr, x, y, cv=5)
        score = scores.mean()
        return score
    from bayes_opt import BayesianOptimization
    rfr_bo = BayesianOptimization(
        rfr_cv,
        {
            'n_estimators': (10, 1000),
            'min_samples_split': (2, 100),
            'max_features': (0.1, 0.999),
            'max_depth': (5, 300)
        }
    )
    rfr_bo.maximize()

    best_max_depth = rfr_bo.res[0]['params']['max_depth']
    best_max_features = rfr_bo.res[0]['params']['max_features']
    best_min_samples_split = rfr_bo.res[0]['params']['min_samples_split']
    best_n_estimators = rfr_bo.res[0]['params']['n_estimators']
    highest_score = rfr_bo.res[0]['target']

    for i in rfr_bo.res:
        if i['target'] > highest_score:
            highest_score = i['target']
            best_max_depth = i['params']['max_depth']
            best_max_features = i['params']['max_features']
            best_min_samples_split = i['params']['min_samples_split']
            best_n_estimators = i['params']['n_estimators']

    # 通过贝叶斯优化找到最佳的参数后，代入模型，模型完成
    rfr = RandomForestRegressor(max_depth=best_max_depth, max_features=best_max_features,
                                min_samples_split=best_min_samples_split.astype(int),
                                n_estimators=best_n_estimators.astype(int))
    return rfr

def dataprepareforcbeforer(data_train,index,deadtime,test_size):
    from sklearn.model_selection import train_test_split
    import copy
    MDs_RT_OSI = data_train[index]  # OSI数据库物质和保留时间相关性大于0.5的分子识别符和保留时间
    MDs_RT_OSI['SMILES'] = data_train['SMILES']  # OSI数据库物质和保留时间相关性大于0.5的分子识别符和保留时间和SMILES
    RT_OSI = data_train['RT']  # OSI数据库的保留时间
    RT_classifier = copy.deepcopy(RT_OSI)  # 用于RandomForestClassifier训练的保留时间
    RT_classifier[RT_classifier <= deadtime] = 0  # 保留时间小于某值的认为为死时间出峰，设置为0
    RT_classifier[RT_classifier > deadtime] = 1  # 保留时间大于某值的认为为非死时间出峰，设置为1
    MDs_RT_OSI['RTC'] = RT_classifier
    MDs_RT_OSI_train, MDs_RT_OSI_test = train_test_split(MDs_RT_OSI,test_size=test_size,random_state=0)
    train_data_classifier = MDs_RT_OSI_train.drop(['RT', 'SMILES', 'RTC'], 1, inplace=False)
    test_data_classifier = MDs_RT_OSI_train['RTC']
    train_target_classifier = MDs_RT_OSI_test.drop(['RT', 'SMILES', 'RTC'], 1, inplace=False)
    test_target_classifier = MDs_RT_OSI_test['RTC']
    return MDs_RT_OSI_train, MDs_RT_OSI_test, train_data_classifier, test_data_classifier, train_target_classifier, test_target_classifier

def classifierbeforeregressor(rfc,train_data,test_data,train_target,MDs_RT_OSI_test):
    rfc.fit(train_data,test_data)
    MDs_RT_OSI_test['Predicted RTC'] = rfc.predict(train_target)
    accurary = len(MDs_RT_OSI_test[MDs_RT_OSI_test['Predicted RTC'] == MDs_RT_OSI_test['RTC']])/len(MDs_RT_OSI_test)
    print(accurary)
    classifier0correct = MDs_RT_OSI_test[MDs_RT_OSI_test['Predicted RTC'] == 0][MDs_RT_OSI_test['RTC'] == 0]
    classifier0correct['Predicted RT'] = classifier0correct['RT']
    classifier0wrong = MDs_RT_OSI_test[MDs_RT_OSI_test['Predicted RTC'] == 0][MDs_RT_OSI_test['RTC'] == 1]
    classifier0wrong['Predicted RT'] = 90
    classifierresult = classifier0correct.append(classifier0wrong)
    classifier1corw = MDs_RT_OSI_test[MDs_RT_OSI_test['Predicted RTC'] == 1]
    return classifierresult, classifier1corw

def dataprepareforrafterc(MDs_RT_OSI_train,classifier1corw,deadtime):
    train_test_data = MDs_RT_OSI_train[MDs_RT_OSI_train['RT'] > deadtime]
    train_data = train_test_data.drop(['RT', 'SMILES', 'RTC'], 1, inplace=False)
    test_data = train_test_data['RT']
    train_target = classifier1corw.drop(['RT', 'SMILES', 'RTC', 'Predicted RTC'], 1, inplace=False)
    test_target = classifier1corw['RT']
    regressordata = classifier1corw
    return train_data, test_data, train_target, test_target, regressordata

def regressorafterclassifier(rfr,train_data,test_data,train_target,test_target,regressordata):
    rfr.fit(train_data, test_data)
    regressordata['Predicted RT'] = rfr.predict(train_target)
    plt.scatter(regressordata['Predicted RTC'], test_target)
    plt.show()
    from sklearn.metrics import r2_score
    candrdata = regressordata.append(classifierresult)
    score = r2_score(candrdata['RT'], candrdata['Predicted RT'])
    print(score)
    return candrdata, score

def publicdbprediction(rfc,rfr,new_numerical,deadtime,data_train,data_test):
    import copy
    target_train_classifier = copy.deepcopy(data_train['RT'])  # OSI数据库的保留时间
    target_train_classifier[target_train_classifier <= deadtime] = 0  # 保留时间小于某值的认为为死时间出峰，设置为0
    target_train_classifier[target_train_classifier > deadtime] = 1  # 保留时间大于某值的认为为非死时间出峰，设置为1
    rfc.fit(data_train[new_numerical], target_train_classifier)
    data_test['Predicted RTC'] = rfc.predict(data_test[new_numerical])
    data_test['Predicted RTC'][data_test['Predicted RTC'] == 0] = 0
    classiferresult = copy.deepcopy(data_test[data_test['Predicted RTC'] == 0])
    test_data = copy.deepcopy(data_test[data_test['Predicted RTC'] == 1])
    data_train_regressor = copy.deepcopy(data_train[index0])
    data_train_regressor = data_train_regressor[data_train_regressor['RT'] > 90]
    rfr.fit(data_train_regressor[new_numerical], data_train_regressor['RT'])
    test_data['Predicted RTC'] = rfr.predict(test_data[new_numerical])
    finalresult = classiferresult.append(test_data)
    return finalresult


'''主程序！！！'''
data_train=readdb(r'F:\Retention time prediction\data', dbname='OSI-SMMS-NEG_MDs.csv')
data_test=readdb(r'F:\Retention time prediction\data', dbname='YL-NEG-20201206_MDs.csv')
miss_data_precent = missvalueprecent(data_train)
miss_data_precent = missvalueprecent(data_test)
data_train, numerical_variable = missvalueprocess(data_train, precent=0.1)
data_test, numerical_variable_nouse = missvalueprocess(data_test, precent=0.1)
new_numerical, index0 = variablefilter(data_train, numerical_variable, relativescore=0.3)
x_classifier, y_classifier = dataprepareforclassifier(data_train, index=index0, deadtime=90)
rfc = deadtimeclassifier(x_classifier, y_classifier)
x_regressor, y_regressor = dataprepareforregressor(data_train,index=index0,deadtime=90)
rfr = trregressor(x_regressor,y_regressor)
MDs_RT_OSI_train, MDs_RT_OSI_test, train_data_classifier, test_data_classifier, train_target_classifier, test_target_classifier = \
    dataprepareforcbeforer(data_train,index=index0,deadtime=90,test_size=0.2)
classifierresult, classifier1corw = classifierbeforeregressor(rfc=rfc,
                                                              train_data=train_data_classifier,
                                                              test_data=test_data_classifier,
                                                              train_target=train_target_classifier,
                                                              MDs_RT_OSI_test=MDs_RT_OSI_test)
train_data_regressor, test_data_regressor, train_target_regressor, test_target_regressor, regressordata = \
    dataprepareforrafterc(MDs_RT_OSI_train,classifier1corw,deadtime=90)
candrdata, finalscore= regressorafterclassifier(rfr,
                                     train_data=train_data_regressor,
                                     test_data=test_data_regressor,
                                     train_target=train_target_regressor,
                                     test_target=test_target_regressor,
                                     regressordata=regressordata)
candrdata.to_csv('加分子指纹+死时间_90+CV_5+变量01+相关系数_00+测试集2_8+测试集预测结果-20201206.csv', index=False)

publicdbpredictionresult = publicdbprediction(rfc,rfr,new_numerical,
                                              deadtime=90,
                                              data_train=MDs_RT_OSI_train,
                                              data_test=MDs_RT_OSI_train)

publicdbpredictionresult.to_csv('加分子指纹+死时间_90+CV_5+变量01+相关系数_00+测试集2_8+训练集预测结果-20201206.csv', index=False)

publicdbpredictionresult = publicdbprediction(rfc,rfr,new_numerical,
                                              deadtime=90,
                                              data_train=data_train,
                                              data_test=data_test)

publicdbpredictionresult.to_csv('YL-NEG-20201206-预测结果.csv', index=True)
