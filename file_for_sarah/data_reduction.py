import matplotlib
import matplotlib.pyplot as plt
#from labellines import labelLine, labelLines
# import seaborn as sns
import os
import pandas as pd
import glob
import numpy as np

results_dir = '/eos/user/s/swaldych/smart_pix/dataset_3s_400NoiseThresh'
dataset_name = 'dataset_3s'
f = open(results_dir+'/final_results.txt', "w+")
f.seek(0)
f.truncate()
# mergePosNeg = pd.concat([big_df_labels, big_df_labels2])
for run_iter in range(4):
    # for sensor_iter in ['50x12P5_0fb', '50x12P5_370fb', '50x12P5_1100fb', '100x25x150_0fb', '100x25x150_370fb', '100x25x150_1100fb']:
    for sensor_iter in ['50x10', '50x12P5', '50x15', '50x20', '50x25', '100x25', '100x25x150']:
        # for thresh_iter in [0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5]:
        for thresh_iter in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            threshold = thresh_iter
            sensor_geom = sensor_iter
            print("=============================")
            print("Analyzing run "+str(run_iter)+": Sensor geometry ",sensor_geom," at pT boundary = ",threshold)
            mergePosNeg = pd.read_csv(results_dir+"/FullTestData_"+sensor_geom+"_0P"+str(threshold - int(threshold))[2:]+"thresh.csv")
            
            # True pT distribution (physical no sign)
            h_physical = plt.hist(abs(mergePosNeg['pt']),bins=np.linspace(0,5,51),histtype='stepfilled');

            # weight per pT bin
            print("Physical hist content: ",h_physical[0])
            w_physical = h_physical[0]/np.sum(h_physical[0])
            
            df1 = pd.read_csv(results_dir+'/TestSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
            df2 = pd.read_csv(results_dir+'/results/'+'/predictionsFiles_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+str(run_iter)+'.csv')
            df3 = pd.read_csv(results_dir+'/results/'+'/testResults_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+str(run_iter)+'.csv')
            df2.columns.values[0] = "predict"
            df3.columns.values[0] = "true"
            df2['predict'] = df2['predict'].astype(int)
            concatenate = pd.concat([df1,df2, df3], axis=1)
            concatenate.head()
            
            # Unphysical pT distribution (no sign)
            h_unphysical = plt.hist(abs(concatenate['pt']),bins=np.linspace(0,5,51),histtype='stepfilled');
            w_unphysical = h_unphysical[0]/np.sum(h_unphysical[0])
            print("Unphysical hist content: ",h_unphysical[0])
            
            # weight each bin up by physical weight, down by unphysical weight
            r = w_physical/w_unphysical
            print("r = ",r)
            
            # Unphysical pT distribution of rejected clusters only
            h_reject = plt.hist(abs(concatenate[concatenate['predict']>0]['pt']),bins=np.linspace(0,5,51))
            np.sum(h_reject[0]*r)/np.sum(h_unphysical[0]*r)
            
            # Unphysical pT distribution of true low pT clusters
            h_lowpt = plt.hist(abs(concatenate[concatenate['true']>0]['pt']),bins=np.linspace(0,5,51))
            np.sum(h_lowpt[0]*r)/np.sum(h_unphysical[0]*r)
            
            # Unphysical pT distribution of true low pT clusters
            h_lowpt = plt.hist(abs(concatenate[abs(concatenate['pt'])<2]['pt']),bins=np.linspace(0,5,51))
            np.sum(h_lowpt[0]*r)/np.sum(h_unphysical[0]*r)
            
            len(mergePosNeg[abs(mergePosNeg['pt'])<2.0])/len(mergePosNeg['pt'])
            
            h_trulyEfficient = plt.hist(abs(concatenate[abs(concatenate['pt'])>2]['pt']),bins=np.linspace(0,5,51),histtype='stepfilled', label='True pT distrib. < 2GeV');
            h_efficiency = plt.hist(abs(concatenate[(abs(concatenate['pt'])>2) & (concatenate['predict']==0)]['pt']),bins=np.linspace(0,5,51), label='Predicted high pT events > 2GeV')
            plt.legend()
            plt.savefig(results_dir+'/data_reduction/'+'sig_eff'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_'+str(run_iter)+'.png')
            sig_eff = np.sum(h_efficiency[0]*r)/np.sum(h_trulyEfficient[0]*r)
            Nse = np.sum(h_trulyEfficient[0])
            print("nse = ",Nse)
            sig_eff_err = np.sqrt(sig_eff*(1-sig_eff)/Nse)
            print("Signal efficiency = {:.2f}%".format(100*np.sum(h_efficiency[0]*r)/np.sum(h_trulyEfficient[0]*r)))
            
            
            h_trulyRejected = plt.hist(abs(concatenate[abs(concatenate['pt'])<2]['pt']),bins=np.linspace(0,5,51),histtype='stepfilled', label='True pT distrib. < 2GeV');
            h_rejected = plt.hist(abs(concatenate[(abs(concatenate['pt'])<2) & (concatenate['predict']>0)]['pt']),bins=np.linspace(0,5,51), label='Predicted high pT events > 2GeV')
            plt.legend()
            plt.savefig(results_dir+'/data_reduction/'+'bkgd_reject'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_'+str(run_iter)+'.png')
            bg_rej = np.sum(h_rejected[0]*r)/np.sum(h_trulyRejected[0]*r)
            Nbg = np.sum(h_trulyRejected[0])
            bg_rej_err = np.sqrt(bg_rej*(1-bg_rej)/Nbg)
            print("Background rejection = {:.2f}%".format(100*np.sum(h_rejected[0]*r)/np.sum(h_trulyRejected[0]*r)))
            
            # Data reduction value
            h_reduction = plt.hist(abs(concatenate[concatenate['predict']>0]['pt']),bins=np.linspace(0,5,51))
            h_unphysical = plt.hist(abs(concatenate['pt']),bins=np.linspace(0,5,51),histtype='stepfilled')
            plt.legend()
            plt.savefig(results_dir+'/data_reduction/'+'data_reduction'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_'+str(run_iter)+'.png')
            dat_red = np.sum(h_reduction[0]*r)/np.sum(h_unphysical[0]*r)
            Ndr = np.sum(h_unphysical[0])
            dat_red_err = np.sqrt(dat_red*(1-dat_red)/Ndr)
            print("Data reduction = {:.2f}%".format(100*np.sum(h_reduction[0]*r)/np.sum(h_unphysical[0]*r)))
            
            with open(results_dir+'/results'+'/final_results.txt', 'a') as file:
                file.write(sensor_geom+','+str(threshold)+','+str(run_iter)+','+str(sig_eff)+','+str(sig_eff_err)+','+str(bg_rej)+','+str(bg_rej_err)+','+ str(dat_red)+','+str(dat_red_err)+'\n')

print("======================")
print("Run complete.")
print("======================")

print("Complete")