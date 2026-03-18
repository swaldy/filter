import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import numpy as np
import csv
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:olive', 'tab:brown']

results_dir = 'results_3s'
dictionary = {
    '50x10': [],
    '50x12P5': [],
    '50x15': [],
    '50x20': [],
    '50x25': [],
    '100x25': [],
    '100x25x150': []
}

legend_dict = {
    '50x10': '50x10x100',
    '50x12P5': '50x12.5x100',
    '50x15': '50x15x100',
    '50x20': '50x20x100',
    '50x25': '50x25x100',
    '100x25': '100x25x100',
    '100x25x150': '100x25x150'
}
# sizes = ['50x10', '50x12P5', '50x15', '50x20', '50x25', '100x25x150']
sizes = ['50x10', '50x12P5', '50x15', '50x20', '50x25', '100x25', '100x25x150']
full_sizes = ['50x10x100', '50x12.5x100', '50x15x100', '50x20x100', '50x25x100', '100x25x100', '100x25x150']
thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
df_temp = pd.read_csv('./'+results_dir+'/final_results.txt',names=["sensor_geom","threshold","run_iter","sig_eff","sig_eff_err","bg_rej","bg_rej_err","dat_red","dat_red_err"])
for size_iter in sizes:
    for threshold in thresholds:
        sig_eff_errStat = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["sig_eff_err"].max()
        sig_eff_errML = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["sig_eff"].std()
        sig_eff_err = np.sqrt(sig_eff_errML**2 + sig_eff_errStat**2)
        sig_eff = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["sig_eff"].mean()
        
        bg_rej_errStat = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["bg_rej_err"].max()
        bg_rej_errML = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["bg_rej"].std()
        bg_rej_err = np.sqrt(bg_rej_errML**2 + bg_rej_errStat**2)
        bg_rej = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["bg_rej"].mean()
        
        dat_red_errStat = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["dat_red_err"].max()
        dat_red_errML = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["dat_red"].std()
        dat_red_err = np.sqrt(dat_red_errML**2 + dat_red_errStat**2)
        dat_red = df_temp[(df_temp.sensor_geom==size_iter) & (df_temp.threshold==threshold)]["dat_red"].mean()
        dictionary[size_iter].append([threshold, sig_eff, sig_eff_err, bg_rej, bg_rej_err, dat_red, dat_red_err])

# with open('./'+results_dir+'/final_results.txt', "r") as file:
#     csvFile = csv.reader(file)
#     for line in csvFile:
#         dictionary[line[0]].append([line[1],line[2], line[3], line[4], line[5], line[6], line[7], line[8]])
#file.write(sensor_geom+','+str(threshold)+','+str(run_iter)+','+str(sig_eff)+','+str(sig_eff_err)+','+str(bg_rej)+','+str(bg_rej_err)+','+ str(dat_red)+','+str(dat_red_err)+'\n')

color_iter = -1
for geom, values in dictionary.items():
    color_iter+=1
    se = []
    se_err = []
    thresh = []
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        se.append(100*float(thresh_iter[1]))
        se_err.append(100*float(thresh_iter[2]))
        if float(thresh_iter[0]) == 0.2:
            print(f"{geom} se = {100*float(thresh_iter[1])} +- {100*float(thresh_iter[2])}")
    geom_tmp = legend_dict[geom]
    if('100x25x150' in geom):
        plt.errorbar(thresh,se,yerr=se_err, capsize=5 ,fmt='-.', color=colors[color_iter])
        plt.scatter(thresh,se, s=20, marker='^', color=colors[color_iter],label=geom_tmp)
    else:
        plt.errorbar(thresh,se,yerr=se_err, capsize=5 ,fmt='-', color=colors[color_iter])
        plt.scatter(thresh,se, s=15, marker='o', color=colors[color_iter],label=geom_tmp)
plt.legend(title="Sensor geometry [um$^3$]", fontsize=12, title_fontsize=12)
plt.xlabel("Low/high pT boundary [GeV]", fontsize=14)
plt.ylabel("Signal efficiency [%]", fontsize=14)
plt.grid()
plt.ylim(70,100)
# plt.show()
plt.tight_layout()
plt.savefig('./'+results_dir+'/final-signal-efficiency.png')
plt.close()

color_iter = -1
for geom, values in dictionary.items():
    color_iter+=1
    bgr = []
    bgr_err = []
    thresh = []
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        bgr.append(100*float(thresh_iter[3]))
        bgr_err.append(100*float(thresh_iter[4]))
    # print(bgr)
    geom_tmp = legend_dict[geom]
    if('100x25x150' in geom):
        plt.errorbar(thresh,bgr,yerr=bgr_err, capsize=5 ,fmt='-.', color=colors[color_iter])
        plt.scatter(thresh,bgr, s=20, marker='^', color=colors[color_iter],label=geom_tmp)
    else:
        plt.errorbar(thresh,bgr,yerr=bgr_err, capsize=5 ,fmt='-', color=colors[color_iter])
        plt.scatter(thresh,bgr, s=15, marker='o', color=colors[color_iter],label=geom_tmp)
plt.legend(title="Sensor geometry [um$^3$]", fontsize=12, title_fontsize=12)
plt.xlabel("Low/high pT boundary [GeV]", fontsize=14)
plt.ylabel("Background rejection [%]", fontsize=14)
plt.grid()
plt.ylim(0,40)
# plt.show()
plt.tight_layout()
plt.savefig('./'+results_dir+'/final-background-rejection.png')
plt.close()

color_iter = -1
for geom, values in dictionary.items():
    color_iter+=1
    # print(geom, values)
    dr = []
    dr_err = []
    thresh = []
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        dr.append(100*float(thresh_iter[5]))
        dr_err.append(100*float(thresh_iter[6]))
        if float(thresh_iter[0]) == 0.2:
            print(f"{geom} dr = {100*float(thresh_iter[5])} +- {100*float(thresh_iter[6])}")
    # print(dr)
    geom_tmp = legend_dict[geom]
    if('100x25x150' in geom):
        plt.errorbar(thresh,dr, yerr=dr_err, capsize=5 ,fmt='-.', color=colors[color_iter])
        plt.scatter(thresh,dr, s=20, marker='^', color=colors[color_iter],label=geom_tmp)
    else:
        plt.errorbar(thresh,dr, yerr=dr_err, capsize=5 ,fmt='-', color=colors[color_iter])
        plt.scatter(thresh,dr, s=15, marker='o', color=colors[color_iter],label=geom_tmp)
plt.legend(title="Sensor geometry [um$^3$]", fontsize=12, title_fontsize=12)
plt.xlabel("Low/high pT boundary [GeV]", fontsize=14)
plt.ylabel("Data reduction [%]", fontsize=14)
plt.grid()
plt.ylim(0,40)
# plt.show()
plt.tight_layout()
plt.savefig('./'+results_dir+'/final-data-reduction.png')
plt.close()

plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    se = []
    se_err = []
    for size_iter in sizes:
        for values in dictionary[size_iter]:
            if values[0] == threshold:
                se.append(100 * values[1])
                se_err.append(100 * values[2])
    # Plot the main line for all but the last two points
    plt.errorbar(full_sizes[:-2], se[:-2], yerr=se_err[:-2], capsize=5, fmt='-', color=colors[threshold_iter])
    # Plot the dotted line for the last two points
    plt.errorbar(full_sizes[-3:], se[-3:], yerr=se_err[-3:], capsize=5, fmt='--', color=colors[threshold_iter])
    # Plot all points as circles
    plt.scatter(full_sizes[:-2], se[:-2], s=15, marker='o', color=colors[threshold_iter], label=f'{threshold}')
    # Plot the last two points as stars
    plt.scatter(full_sizes[-2:], se[-2:], s=50, marker='*', color=colors[threshold_iter])

plt.xlabel('Sensor Geometry [um$^3$]', fontsize=14)
plt.ylabel('Signal Efficiency (%)', fontsize=14)
plt.legend(title="Low/high p$_T$ \nboundary [GeV]", fontsize=12, title_fontsize=12)
plt.ylim(70,100)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'./{results_dir}/final-signal-efficiency-vs-geometry.png')
plt.close()

plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    bgr = []
    bgr_err = []
    for size_iter in sizes:
        for values in dictionary[size_iter]:
            if values[0] == threshold:
                bgr.append(100 * values[3])
                bgr_err.append(100 * values[4])
    # Plot the main line for all but the last two points
    plt.errorbar(full_sizes[:-2], bgr[:-2], yerr=bgr_err[:-2], capsize=5, fmt='-', color=colors[threshold_iter])
    # Plot the dotted line for the last two points
    plt.errorbar(full_sizes[-3:], bgr[-3:], yerr=bgr_err[-3:], capsize=5, fmt='--', color=colors[threshold_iter])
    # Plot all points as circles
    plt.scatter(full_sizes[:-2], bgr[:-2], s=15, marker='o', color=colors[threshold_iter], label=f'{threshold}')
    # Plot the last two points as stars
    plt.scatter(full_sizes[-2:], bgr[-2:], s=50, marker='*', color=colors[threshold_iter])

plt.xlabel('Sensor Geometry [um$^3$]', fontsize=14)
plt.ylabel('Background rejection (%)', fontsize=14)
plt.legend(title="Low/high p$_T$ \nboundary [GeV]",loc='center left', fontsize=12, title_fontsize=12, bbox_to_anchor=(0.57, 0.72))
plt.ylim(10,70)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'./{results_dir}/final-background-rejection-vs-geometry.png')
plt.close()

plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    dr = []
    dr_err = []
    for size_iter in sizes:
        for values in dictionary[size_iter]:
            if values[0] == threshold:
                dr.append(100 * values[5])
                dr_err.append(100 * values[6])
    # Plot the main line for all but the last two points
    plt.errorbar(full_sizes[:-2], dr[:-2], yerr=dr_err[:-2], capsize=5, fmt='-', color=colors[threshold_iter])
    # Plot the dotted line for the last two points
    plt.errorbar(full_sizes[-3:], dr[-3:], yerr=dr_err[-3:], capsize=5, fmt='--', color=colors[threshold_iter])
    # Plot all points as circles
    plt.scatter(full_sizes[:-2], dr[:-2], s=15, marker='o', color=colors[threshold_iter], label=f'{threshold}')
    # Plot the last two points as stars
    plt.scatter(full_sizes[-2:], dr[-2:], s=50, marker='*', color=colors[threshold_iter])

plt.xlabel('Sensor Geometry [um$^3$]')
plt.ylabel("Data reduction [%]", fontsize=14)
plt.legend(title="Low/high p$_T$ \nboundary [GeV]",loc='center left', bbox_to_anchor=(0.58, 0.75))
plt.ylim(10,60)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'./{results_dir}/final-data-reduction-vs-geometry.png')