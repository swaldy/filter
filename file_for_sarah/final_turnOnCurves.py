import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
pd.options.display.max_seq_items = 50
# import seaborn as sns

dataset_name = 'dataset_7s'
results_dir = 'results_7s'
models_dir = 'models_7s'
run_num = '9'
sizes = ['50x10', '50x12P5', '50x15', '50x20', '50x25', '100x25', '100x25x150']
# dataset_name = 'dataset_9s_400NoiseThresh'
# results_dir = 'results_9s_400NoiseThresh_2s_trained'
# models_dir = 'models_9s_400NoiseThresh'
# run_num='3'
# sizes = ['50x12P5_0fb', '50x12P5_370fb', '50x12P5_1100fb']
thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

# Negative side
for sensor_geom in sizes:
    for threshold in thresholds:
        print("=============================")
        print("Producing results for ",sensor_geom," at pT boundary = ",threshold)
        df1 = pd.read_csv('./'+dataset_name+'/TestSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
        df2 = pd.read_csv('./'+results_dir+'/predictionsFiles_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+run_num+'.csv')
        df3 = pd.read_csv('./'+results_dir+'/testResults_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+run_num+'.csv')
        df2.columns.values[0] = "predict"
        df3.columns.values[0] = "true"
        df2['predict'] = df2['predict'].astype(int)
        concatenate = pd.concat([df1,df2, df3], axis=1)
        list1 = []
        list2 = []
        xvalues = [0.0, -0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7,-1.8,-1.9,-2.0,-2.5,-3.0,-3.5,-4.0,-4.5,-5.0]
        
        for i in range(len(xvalues)-1):
            binValidate = concatenate.loc[(concatenate['pt'] > xvalues[i+1]) & (concatenate['pt'] <= xvalues[i])]
            list1.append(binValidate.shape[0])
            list2.append(sum((binValidate.predict == 0)) / (binValidate.shape[0]))
        
        reversed_list2 = list2[::-1]
        #reversed_list2
        print("Length of reversed list = ",len(reversed_list2))
        reversed_list = list1[::-1]
        y_values=np.array([reversed_list2])
        x_values=np.array([reversed_list])
        first = y_values*(1-y_values)
        second = first/x_values
        errors = np.sqrt(second)
        print("X values = ",x_values)
        print("Y values = ",y_values)
        print("Errors = ",errors)
        np.savetxt('./'+results_dir+'/NegativeYValuesRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh2_run'+run_num+'.out',y_values, delimiter=',')
        np.savetxt('./'+results_dir+'/errorsNegativeRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh2_run'+run_num+'.out', errors, delimiter=',')

        # Positive side
for sensor_geom in sizes:
    for threshold in thresholds:
        print("=============================")
        print("Producing results for ",sensor_geom," at pT boundary = ",threshold)
        df1 = pd.read_csv('./'+dataset_name+'/TestSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
        df2 = pd.read_csv('./'+results_dir+'/predictionsFiles_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+run_num+'.csv')
        df3 = pd.read_csv('./'+results_dir+'/testResults_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+run_num+'.csv')
        df2.columns.values[0] = "predict"
        df3.columns.values[0] = "true"
        df2['predict'] = df2['predict'].astype(int)
        concatenate = pd.concat([df1,df2, df3], axis=1)
        list1 = []
        list2 = []
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for i in range(len(bin_edges) - 1):
            bin = concatenate.loc[(concatenate['pt'] < bin_edges[i+1]) & (concatenate['pt'] >= bin_edges[i])]
            list1.append(bin.shape[0])
            list2.append(sum(bin.predict == 0) / bin.shape[0])
        
        # %%
        y_values = np.array([list2])
        x_values = np.array([list1])
        errors = np.sqrt(y_values * (1 - y_values) / x_values)
        print("X values = ",x_values)
        print("Y values = ",y_values)
        print("Errors = ",errors)
        np.savetxt('./'+results_dir+'/PositiveYValuesRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh2_run'+run_num+'.out', y_values, delimiter=',')
        np.savetxt('./'+results_dir+'/errorsPositiveRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh2_run'+run_num+'.out', errors, delimiter=',')

prefix = "2"
xvalues = [-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2, -0.1, 0, 0, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.5,3.0,3.5,4.0,4.5]
thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
for sensor_geom in sizes:
    for i, iter in enumerate(thresholds):
        print("=============================")
        print("Producing results for ",sensor_geom," at pT boundary = ",threshold)
        threshold = iter
        df1 = pd.read_csv('./'+results_dir+'/PositiveYValuesRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh'+prefix+'_run'+run_num+'.out', header=None)
        df1 = df1.T
        df2 = pd.read_csv('./'+results_dir+'/NegativeYValuesRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh'+prefix+'_run'+run_num+'.out', header=None)
        df2 = df2.T
        df1.columns =['yvalues']
        df2.columns =['yvalues']
        df3 = pd.concat([df2,df1],join="inner")
        
        df1_error = pd.read_csv('./'+results_dir+'/errorsPositiveRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh'+prefix+'_run'+run_num+'.out', header=None)
        df1_error = df1_error.T
        df2_error = pd.read_csv('./'+results_dir+'/errorsNegativeRebin_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh'+prefix+'_run'+run_num+'.out', header=None)
        df2_error = df2_error.T
        df1_error.columns =['error']
        df2_error.columns =['error']
        df3_error = pd.concat([df2_error,df1_error],join="inner")
        
        mergeDF = pd.concat([df3,df3_error],axis=1)
        mergeDF['xvalues'] = xvalues
        xvaluesArray = np.array(mergeDF['xvalues'])
        yvaluesArray = np.array(mergeDF['yvalues'])
        errorArray = np.array(mergeDF['error'])
        plt.errorbar(xvaluesArray, yvaluesArray, yerr=errorArray, label="Boundary="+str(threshold))
        # plt.scatter(xvaluesArray, yvaluesArray)
    
    plt.legend()
    if sensor_geom=='50x12P5':
        geom_tmp = '50x12.5x100'
    elif '150' not in sensor_geom:
        geom_tmp = sensor_geom+'x100'
    plt.title('Turn-on curve for '+geom_tmp+' um3 sensor')
    plt.ylim([0,1.01])
    plt.xlabel(r'true $P_{T}$ (GeV)')
    plt.ylabel('fraction classified as > abs (low/high pT boundary)')
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # plt.xticks([-2,-1.5,-1,-0.5,-0.4,-0.3,-0.2, -0.1, 0, 0.1,0.2,0.3,0.4,0.5,1.0,1.5,2], rotation=45)
    plt.xticks([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2,-1.5,-1,-0.6,-0.3,0, 0.3,0.6,1.0,1.5,2,2.5,3.0,3.5,4.0,4.5,5.0], rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig('./'+results_dir+'/multiclassifier_AllThresh_'+sensor_geom+'_'+prefix+'_run'+run_num+'.png')
    plt.close()