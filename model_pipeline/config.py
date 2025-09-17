# ================================
#           TEST BENCH 1
# ================================
# Run 1
confs_TB1_ABtuning = [
    {
        "qm_charge_levels" : [400, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.10_10.36.52_DNN_vth0-0.020_vth1-0.085_vth2-0.135/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.10_10.36.52_DNN_vth0-0.020_vth1-0.085_vth2-0.135/yprofiles.csv"
        },
    # {
    #     "qm_charge_levels" : [700, 1600, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.15_00.00.26_DNN_vth0-0.031_vth1-0.085_vth2-0.135/final_results.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.15_00.00.26_DNN_vth0-0.031_vth1-0.085_vth2-0.135/yprofiles.csv"
    #     },
    # {
    #     "qm_charge_levels" : [1000, 1600, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.11_10.05.20_DNN_vth0-0.050_vth1-0.085_vth2-0.135/final_results.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.11_10.05.20_DNN_vth0-0.050_vth1-0.085_vth2-0.135/yprofiles.csv"
    #     },
    # {
    #     "qm_charge_levels" : [1200, 1600, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.20_00.05.10_DNN_vth0-0.059_vth1-0.085_vth2-0.135/final_results.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.20_00.05.10_DNN_vth0-0.059_vth1-0.085_vth2-0.135/yprofiles.csv"
    #     },
    {
        "qm_charge_levels" : [1300, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.22_01.07.20_DNN_vth0-0.064_vth1-0.085_vth2-0.135/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.22_01.07.20_DNN_vth0-0.064_vth1-0.085_vth2-0.135/yprofiles.csv"
        },
    # {
    #     "qm_charge_levels" : [1600, 2000, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.25_23.33.16_DNN_vth0-0.085_vth1-0.105_vth2-0.135/final_results.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.25_23.33.16_DNN_vth0-0.085_vth1-0.105_vth2-0.135/yprofiles.csv"
    #     },
    
]

# Run 2 - low match with RTL
confs_TB1_BDtuning2 = [
    {
        "qm_charge_levels" : [400, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.01_00.00.06_DNN_vth0-0.014_vth1-0.083_vth2-0.128/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.01_00.00.06_DNN_vth0-0.014_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
    {
        "qm_charge_levels" : [1300, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.30_01.31.47_DNN_vth0-0.064_vth1-0.085_vth2-0.135/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.08.30_01.31.47_DNN_vth0-0.064_vth1-0.085_vth2-0.135/yprofiles.csv"
        },

    {
        "qm_charge_levels" : [1600, 2000, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.03_07.35.59_DNN_vth0-0.037_vth1-0.083_vth2-0.128/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.03_07.35.59_DNN_vth0-0.037_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
        
]

# Run 3 - 97% match with RTL
confs_TB1_BDtuning3 = [
    {
        "qm_charge_levels" : [400, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.07_19.53.03_DNN_vth0-0.014_vth1-0.083_vth2-0.128/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.07_19.53.03_DNN_vth0-0.014_vth1-0.083_vth2-0.128/yprofiles.csv"
        }
        
]

# Run 4 - 99.86% match with RTL at timestamp 18 (best RTL match was obtained on timestamp 18 while using dataset 8)
confs_TB1_BDtuning4_TS18 = [
    {
        "qm_charge_levels" : [400, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.10_14.42.55_DNN_vth0-0.014_vth1-0.083_vth2-0.128/final_results_ts18.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.10_14.42.55_DNN_vth0-0.014_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
    {
        "qm_charge_levels" : [1000, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.09_18.06.40_DNN_vth0-0.048_vth1-0.083_vth2-0.128/final_results_ts18.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.09_18.06.40_DNN_vth0-0.048_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
        
]

# Run 4 - 99.21% match with RTL at timestamp 20 (best RTL match was obtained on timestamp 20 while using dataset 14)
confs_TB1_BDtuning4_TS20 = [
    # {
    #     "qm_charge_levels" : [400, 1600, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.10_14.42.55_DNN_vth0-0.014_vth1-0.083_vth2-0.128/final_results_ts20.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.10_14.42.55_DNN_vth0-0.014_vth1-0.083_vth2-0.128/yprofiles.csv"
    #     },
    {
        "qm_charge_levels" : [700, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.11_22.36.24_DNN_vth0-0.031_vth1-0.083_vth2-0.128/final_results_ts20.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.11_22.36.24_DNN_vth0-0.031_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
    {
        "qm_charge_levels" : [1000, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.09_18.06.40_DNN_vth0-0.048_vth1-0.083_vth2-0.128/final_results_ts20.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.09_18.06.40_DNN_vth0-0.048_vth1-0.083_vth2-0.128/yprofiles.csv"
        },
    {
        "qm_charge_levels" : [1600, 2000, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.15_02.40.00_DNN_vth0-0.083_vth1-0.106_vth2-0.128/final_results_ts20.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.15_02.40.00_DNN_vth0-0.083_vth1-0.106_vth2-0.128/yprofiles.csv"
        },
]

# ================================
#           TEST BENCH 2
# ================================
confs_TB2_BDtuning = [
    # {
    #     "qm_charge_levels" : [400, 1600, 2400], 
    #     "asic_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.01_00.00.06_DNN_vth0-0.014_vth1-0.083_vth2-0.128/final_results.npy",
    #     "asic_yprofiles_path" : "/mnt/local/CMSPIX28/data/ChipVersion1_ChipID17_SuperPix1/2025.09.01_00.00.06_DNN_vth0-0.014_vth1-0.083_vth2-0.128/yprofiles.csv"
    #     },
    {
        "qm_charge_levels" : [1300, 1600, 2400], 
        "asic_path" : "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID18_SuperPix1/2025.08.30_00.40.46_DNN_vth0-0.064_vth1-0.085_vth2-0.135/final_results.npy",
        "asic_yprofiles_path" : "/mnt/local/CMSPIX28/Scurve/data/ChipVersion1_ChipID18_SuperPix1/2025.08.30_00.40.46_DNN_vth0-0.064_vth1-0.085_vth2-0.135/yprofiles.csv"
        },
        
]

