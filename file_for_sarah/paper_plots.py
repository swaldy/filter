import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import numpy as np
import csv
from matplotlib.lines import Line2D

colors_dict={'50x10': (128/255.,128/255.,128/255.), 
             '50x12P5': (51/255.,187/255.,238/255.),
             '50x15': (204/255.,102/255.,119/255.),
             '50x20': (153/255.,153/255.,51/255.),
             '50x25': (17/255.,119/255.,51/255.),
             '100x25': (136/255.,34/255.,85/255.),
             '100x25x150': (51/255.,34/255.,136/255.)
            }
markers_dict = {
    '50x10': ['-', 'v', 40], 
    '50x12P5': ['-', '^', 30],
    '50x15': ['-', '<', 40],
    '50x20': ['-', '>', 40],
    '50x25': ['-', '*', 40],
    '100x25': ['--', 'o', 40],
    '100x25x150': [':', 'o', 40]
}

colors_dict_old={'50x10': 'tab:brown', 
             '50x12P5': 'tab:red',
             '50x15': 'tab:green',
             '50x20': 'tab:orange',
             '50x25': 'tab:purple',
             '100x25': 'tab:olive',
             '100x25x150': 'tab:blue'
            }
results_dir = 'results_3s'
save_dir = 'paper_plots'

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

# Paper cosmetic changes (4July25)
# dontplot = []
dontplot = ['50x10', '50x15', '50x20']

# After plotting all your data, define custom legend handles:
legend_handles = []
legend_labels = []

for geom in [g for g in dictionary.keys() if g not in dontplot]:
    color = colors_dict[geom]
    label = legend_dict[geom]
    [fmt, marker, ms] = markers_dict[geom]
    if '100x25' in geom:
        markerface='none'
    else:
        markerface=color

    # Create a dummy errorbar for the legend only
    handle = plt.errorbar([0.1], [0.1], yerr=[[0.5], [0.5]], fmt=fmt, color=color,
                          marker=marker, markerfacecolor=markerface, markersize=9, capsize=5)
    legend_handles.append(handle)
    legend_labels.append(label)

for geom, values in dictionary.items():
    se = []
    se_err = []
    thresh = []
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        se.append(100*float(thresh_iter[1]))
        se_err.append(100*float(thresh_iter[2]))
        if float(thresh_iter[0]) == 0.2:
            print(f"{geom} se = {100*float(thresh_iter[1])} +- {100*float(thresh_iter[2])}")
    # print(se)
    geom_tmp = legend_dict[geom]
    [fmt, marker, ms] = markers_dict[geom]
    if(geom in dontplot):
        continue
    if('100x25x150' in geom) or ('100x25' in geom):
        plt.errorbar(thresh,se,yerr=se_err, capsize=5 ,fmt=fmt, color=colors_dict[geom],label=geom_tmp)
        plt.scatter(thresh,se, s=ms, marker=marker, facecolors='none', color=colors_dict[geom])
        # plt.scatter(thresh, se, s=30, marker='s', facecolors='none', edgecolors=colors_dict[geom])
    else:
        plt.errorbar(thresh,se,yerr=se_err, capsize=5 ,fmt=fmt, color=colors_dict[geom],label=geom_tmp)
        plt.scatter(thresh,se, s=ms, marker=marker, color=colors_dict[geom])
# plt.legend(title="Sensor geometry [$\mu m^3$]", fontsize=12, title_fontsize=12)
plt.legend([h for h in legend_handles], legend_labels,
           title="Sensor geometry [$\mu m^3$]", fontsize=14, title_fontsize=14)
plt.xlabel("p$_T$ boundary [GeV]", fontsize=16)
plt.ylabel("Signal efficiency [%]", fontsize=16)
# plt.grid()
# ylimits = [(25,100), (0,75), (0,75)]
plt.ylim(25,100)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
plt.savefig('./'+save_dir+'/final-signal-efficiency-vs-geometry-allpt.png', dpi=300)
plt.close()

for geom, values in dictionary.items():
    bgr = []
    bgr_err = []
    thresh = []
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        bgr.append(100*float(thresh_iter[3]))
        bgr_err.append(100*float(thresh_iter[4]))
    # print(bgr)
    geom_tmp = legend_dict[geom]
    [fmt, marker, ms] = markers_dict[geom]
    if(geom in dontplot):
        continue
    if('100x25x150' in geom) or ('100x25' in geom):
        plt.errorbar(thresh,bgr,yerr=bgr_err, capsize=5 ,fmt=fmt, color=colors_dict[geom],label=geom_tmp)
        plt.scatter(thresh,bgr, s=ms, marker=marker, facecolors='none', color=colors_dict[geom])
    else:
        plt.errorbar(thresh,bgr,yerr=bgr_err, capsize=5 ,fmt=fmt, color=colors_dict[geom],label=geom_tmp)
        plt.scatter(thresh,bgr, s=ms, marker=marker, color=colors_dict[geom])
# plt.legend(title="Sensor geometry [$\mu m^3$]", fontsize=12, title_fontsize=12)
plt.legend([h for h in legend_handles], legend_labels,
           title="Sensor geometry [$\mu m^3$]", fontsize=14, title_fontsize=14)
plt.xlabel("p$_T$ boundary [GeV]", fontsize=16)
plt.ylabel("Background rejection [%]", fontsize=16)
# plt.grid()
plt.ylim(0,75)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
plt.savefig('./'+save_dir+'/final-background-rejection-vs-geometry-allpt.png', dpi=300)
plt.close()

for geom, values in dictionary.items():
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
    [fmt, marker, ms] = markers_dict[geom]
    if(geom in dontplot):
        continue
    if('100x25x150' in geom) or ('100x25' in geom):
        plt.errorbar(thresh,dr, yerr=dr_err, capsize=5 ,fmt=fmt, color=colors_dict[geom])
        plt.scatter(thresh,dr, s=ms, marker=marker, facecolors='none', color=colors_dict[geom],label=geom_tmp)
    else:
        plt.errorbar(thresh,dr, yerr=dr_err, capsize=5 ,fmt=fmt, color=colors_dict[geom])
        plt.scatter(thresh,dr, s=ms, marker=marker, color=colors_dict[geom],label=geom_tmp)
# plt.legend(title="Sensor geometry [$\mu m^3$]", fontsize=12, title_fontsize=12)
plt.legend([h for h in legend_handles], legend_labels,
           title="Sensor geometry [$\mu m^3$]", fontsize=14, title_fontsize=14)
plt.xlabel("p$_T$ boundary [GeV]", fontsize=16)
plt.ylabel("Data reduction [%]", fontsize=16)
# plt.grid()
plt.ylim(0,75)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
plt.savefig('./'+save_dir+'/final-data-reduction-vs-geometry-allpt.png', dpi=300)
plt.show()
plt.close()

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

save_dir = './paper_plots/'
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

thresh = []
se = []
se_err = []
bgr = []
bgr_err = []
dr = []
dr_err = []
for geom, values in dictionary.items():
    for thresh_iter in values:
        thresh.append(float(thresh_iter[0]))
        se.append(100*float(thresh_iter[1]))
        se_err.append(100*float(thresh_iter[2]))
        bgr.append(100*float(thresh_iter[3]))
        bgr_err.append(100*float(thresh_iter[4]))
        dr.append(100*float(thresh_iter[5]))
        dr_err.append(100*float(thresh_iter[6]))
plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    if threshold == 0.2:
        se = []
        se_err = []
        for size_iter in sizes:
            for values in dictionary[size_iter]:
                if values[0] == threshold:
                    se.append(100 * values[1])
                    se_err.append(100 * values[2])
        # Plot the main line for all but the last three points
        plt.errorbar(full_sizes[:-2], se[:-2], yerr=se_err[:-2], capsize=5, fmt='-', color='k')
        # Plot the dashed line for the third and second last points
        plt.errorbar(full_sizes[-3:-1], se[-3:-1], yerr=se_err[-3:-1], capsize=5, fmt='--', color='k')
        # Plot the dotted line for the last two points
        plt.errorbar(full_sizes[-2:], se[-2:], yerr=se_err[-2:], capsize=5, fmt=':', color='k')
        for dict_iter, (geom, values) in enumerate(dictionary.items()):
            [fmt, marker, ms] = markers_dict[geom]
            if '100x25' in geom:
                markerface='none'
            else:
                markerface='k'
            plt.scatter(full_sizes[dict_iter], se[dict_iter], s=ms, marker=marker, facecolors=markerface, color='k', label=f'{threshold}')
        # # Plot all points as circles
        # plt.scatter(full_sizes[:-2], se[:-2], s=15, marker='o', color='k', label=f'{threshold}')
        # # Plot the second last point as a diamond
        # plt.scatter(full_sizes[-2:-1], se[-2:-1], s=40, marker='o', facecolors='none', color='k')
        # # Plot the last point as a star
        # plt.scatter(full_sizes[-1:], se[-1:], s=100, marker='*', color='k')

plt.xlabel('Sensor geometry [$\mu m^3$]', fontsize=16)
plt.ylabel('Signal efficiency (%)', fontsize=16)
# plt.legend(title="Low/high p$_T$ \nboundary [GeV]", fontsize=12, title_fontsize=12)
# ylimits = [(25,100), (0,75), (0,75)]
plt.ylim(25,100)
# plt.grid(True)
plt.yticks(fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.savefig(f'./{save_dir}/final-signal-efficiency-vs-geometry.png', dpi=300)
plt.close()

plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    if threshold == 0.2:
        bgr = []
        bgr_err = []
        for size_iter in sizes:
            for values in dictionary[size_iter]:
                if values[0] == threshold:
                    bgr.append(100 * values[3])
                    bgr_err.append(100 * values[4])
        # Plot the main line for all but the last three points
        plt.errorbar(full_sizes[:-2], bgr[:-2], yerr=bgr_err[:-2], capsize=5, fmt='-', color='k')
        # Plot the dashed line for the third and second last points
        plt.errorbar(full_sizes[-3:-1], bgr[-3:-1], yerr=bgr_err[-3:-1], capsize=5, fmt='--', color='k')
        # Plot the dotted line for the last two points
        plt.errorbar(full_sizes[-2:], bgr[-2:], yerr=bgr_err[-2:], capsize=5, fmt=':', color='k')
        for dict_iter, (geom, values) in enumerate(dictionary.items()):
            [fmt, marker, ms] = markers_dict[geom]
            if '100x25' in geom:
                markerface='none'
            else:
                markerface='k'
            plt.scatter(full_sizes[dict_iter], bgr[dict_iter], s=ms, marker=marker, facecolors=markerface, color='k', label=f'{threshold}')
        # # Plot all points as circles
        # plt.scatter(full_sizes[:-2], bgr[:-2], s=15, marker='o', color='k', label=f'{threshold}')
        # # Plot the second last point as a diamond
        # plt.scatter(full_sizes[-2:-1], bgr[-2:-1], s=40, marker='o', facecolors='none',  color='k')
        # # Plot the last point as a star
        # plt.scatter(full_sizes[-1:], bgr[-1:], s=100, marker='*', color='k')
        

plt.xlabel('Sensor geometry [$\mu m^3$]', fontsize=16)
plt.ylabel('Background rejection (%)', fontsize=16)
# plt.legend(title="Low/high p$_T$ \nboundary [GeV]",loc='center left', fontsize=12, title_fontsize=12, bbox_to_anchor=(0.57, 0.72))
plt.ylim(0,75)
# plt.grid(True)
plt.yticks(fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.savefig(f'./{save_dir}/final-background-rejection-vs-geometry.png', dpi=300)
plt.close()

plt.figure()
for threshold_iter, threshold in enumerate(thresholds):
    if threshold == 0.2:
        dr = []
        dr_err = []
        for size_iter in sizes:
            for values in dictionary[size_iter]:
                if values[0] == threshold:
                    dr.append(100 * values[5])
                    dr_err.append(100 * values[6])
        # Plot the main line for all but the last three points
        plt.errorbar(full_sizes[:-2], dr[:-2], yerr=dr_err[:-2], capsize=5, fmt='-', color='k')
        # Plot the dashed line for the third and second last points
        plt.errorbar(full_sizes[-3:-1], dr[-3:-1], yerr=dr_err[-3:-1], capsize=5, fmt='--', color='k')
        # Plot the dotted line for the last two points
        plt.errorbar(full_sizes[-2:], dr[-2:], yerr=dr_err[-2:], capsize=5, fmt=':', color='k')
        for dict_iter, (geom, values) in enumerate(dictionary.items()):
            [fmt, marker, ms] = markers_dict[geom]
            if '100x25' in geom:
                markerface='none'
            else:
                markerface='k'
            plt.scatter(full_sizes[dict_iter], dr[dict_iter], s=ms, marker=marker, facecolors=markerface, color='k', label=f'{threshold}')
        # # Plot all points as circles
        # plt.scatter(full_sizes[:-2], dr[:-2], s=15, marker='o', color='k', label=f'{threshold}')
        # # Plot the second last point as a diamond
        # plt.scatter(full_sizes[-2:-1], dr[-2:-1], s=40, marker='o', facecolors='none', color='k')
        # # Plot the last point as a star
        # plt.scatter(full_sizes[-1:], dr[-1:], s=100, marker='*', color='k')

plt.xlabel('Sensor geometry [$\mu m^3$]', fontsize=16)
plt.ylabel("Data reduction [%]", fontsize=16)
# plt.legend(title="Low/high p$_T$ \nboundary [GeV]",loc='center left', bbox_to_anchor=(0.58, 0.75))
plt.ylim(0,75)
# plt.grid(True)
plt.yticks(fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.savefig(f'./{save_dir}/final-data-reduction-vs-geometry.png', dpi=300)