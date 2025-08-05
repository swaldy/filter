try:
    import os
    import sys
    import glob
    import pandas as pd
    import numpy as np
    import hls4ml
    from fxpmath import Fxp
    import csv
    from tqdm import tqdm

    # custom imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pretrain-data-prep')) # https://github.com/smart-pix/pretrain-data-prep/tree/main
    from dataset_utils import quantize_manual

except:
    print("Import error:", f"{__file__}: {str(e)}")
    sys.exit(1)  # Exit script immediately
    
# load example inputs and outputs
def loadExampleTestVectors():
    x_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/input_1.csv", header=None)
    x_test = np.array(x_test.values.tolist())
    y_test = pd.read_csv("/asic/projects/C/CMS_PIX_28/benjamin/verilog/workarea/cms28_smartpix_verification/PnR_cms28_smartpix_verification_D/tb/dnn/csv/l6/layer7_out_ref_int.csv", header=None)
    y_test = np.array(y_test.values.tolist()).flatten()
    return x_test, y_test

# load data from parquet files
def loadParquetData(
    inFilePath="data/", # in path where the labels, recon2D files sit
    noise_threshold = 0, # thresholds on the number of electrons
    threshold = 0.2, # pt threshold in GeV for high pT particle
    qm_charge_levels = [400, 1600, 2400], # quantize manual charge levels
    qm_quant_values = [0, 1, 2, 3], # quantize manual quant values
):
    
    # load the labels and data
    inFilePaths = list(sorted(glob.glob(os.path.join(inFilePath, "labels*")))) 
    trainlabels, trainrecons = [], []
    for inFiles in tqdm(inFilePaths):
        trainlabels.append(pd.read_parquet(inFiles))
        temp = pd.read_parquet(inFiles.replace("labels", "recon2D"))
        temp = quantize_manual(temp, charge_levels=qm_charge_levels, quant_values=qm_quant_values)
        trainrecons.append(temp)

    # convert to csv
    trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
    trainrecons_csv = pd.concat(trainrecons, ignore_index=True)
    print(len(trainlabels_csv), len(trainrecons_csv))

    # function to sum over the x rows to create the y-profile
    def sumRow(X):
        X = np.where(X < noise_threshold, 0, X)
        sum1 = 0
        sumList = []
        for i in X:
            sum1 = np.sum(i,axis=0)
            sumList.append(sum1)
            b = np.array(sumList)
        return b
    
    # create the trainlist1 = yprofiles, trainlist2 = y-local, cls, pt
    print("Creating yprofiles")
    yprofiles, ylocals, clslabels = [], [], []
    for (index1, row1), (index2, row2) in zip(trainrecons_csv.iterrows(), trainlabels_csv.iterrows()):
        rowSum = 0.0
        X = row1.values
        X = np.reshape(X,(13,21))
        rowSum = sumRow(X)
        yprofiles.append(rowSum) 
        cls = -1
        if(abs(row2['pt'])>threshold):
            cls=0
        elif(-1*threshold<=row2['pt']<0):
            cls=1
        elif(0<=row2['pt']<=threshold):
            cls=2
        ylocals.append(row2["y-local"])
        clslabels.append(cls)

    # create numpys
    yprofiles = np.array(yprofiles)
    ylocals = np.array(ylocals)
    clslabels = np.array(clslabels)

    # pad the yprofiles to get to 16 dimension
    yprofiles = np.pad(yprofiles, ((0, 0), (0, 3)), mode='constant', constant_values=0)

    return yprofiles, ylocals, clslabels

# Generate a simple configuration from keras model
def gen_hls_model(model, output_dir="newModelWeights"):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    # Convert to an hls model
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)
    hls_model.write()

# convert code from hsl4ml style output to chip style input
def prepareWeights(path):
    data_fxp = Fxp(None, signed=True, n_word=4, n_int=0)
    data_fxp.rounding = 'around'
    def to_fxp(val):
        return data_fxp(val)

    b5_data = pd.read_csv(os.path.join(path, 'b5.txt'), header=None)
    w5_data = pd.read_csv(os.path.join(path, 'w5.txt'), header=None)
    b2_data = pd.read_csv(os.path.join(path, 'b2.txt'), header=None)
    w2_data = pd.read_csv(os.path.join(path, 'w2.txt'), header=None)
    # print(b5_data)

    b5_data_list = []
    w5_data_list = []
    b2_data_list = []
    w2_data_list = []

    for i in range(2, -1, -1):
        b5_data_list.append(to_fxp(b5_data.values[0][i]).bin())

    for i in range(173, -1, -1):
        w5_data_list.append(to_fxp(w5_data.values[0][i]).bin())

    for i in range(57, -1, -1):
        b2_data_list.append(to_fxp(b2_data.values[0][i]).bin())

    for i in range(927, -1, -1):
        w2_data_list.append(to_fxp(w2_data.values[0][i]).bin())

    b5_bin_list = [int(bin_string) for data in b5_data_list for bin_string in data]
    w5_bin_list = [int(bin_string) for data in w5_data_list for bin_string in data]
    b2_bin_list = [int(bin_string) for data in b2_data_list for bin_string in data]
    w2_bin_list = [int(bin_string) for data in w2_data_list for bin_string in data]
    pixel_list = [0 for _ in range(512)]
    b5_w5_b2_w2_pixel_list = b5_bin_list + w5_bin_list + b2_bin_list + w2_bin_list + pixel_list

    csv_file = os.path.join(path, 'b5_w5_b2_w2_pixel_bin.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(b5_w5_b2_w2_pixel_list)
    
    return csv_file

# convert yprofiles to the pixel programming for asic
def input_to_pixelout(x):
    N_INFERENCES = x.shape[0]

    # first create compout
    encoder_values_Ninferences = []
    for i in range(N_INFERENCES):
        encoder_values = []
        for j in range(16):
            a = x[i][j]
            encoder_sum = []
            if(a==0):
                encoder_sum = [0 for _ in range(16)]
                encoder_sum.reverse()
                
            elif(a==1):
                encoder_sum = [1]+[0 for _ in range(15)]
                encoder_sum.reverse()

            elif(a==2):
                encoder_sum = [2]+[0 for _ in range(15)]
                encoder_sum.reverse()

            elif(a==3):
                encoder_sum = [3]+[0 for _ in range(15)]
                encoder_sum.reverse()

            else:
                l3=[]
                if a%3 == 0:
                    result = int(a/3)
                    l3 = [3 for _ in range(result)]
                    l_diff = 16-len(l3)
                    encoder_sum = l3 + [0 for _ in range(l_diff)]
                    encoder_sum.reverse()
                else:
                    result = int(a//3)
                    l3 = [3 for _ in range(result)]
                    diff = a-sum(l3)
                    l3.append(diff)
                    l_diff = 16-len(l3)
                    encoder_sum = l3 + [0 for _ in range(l_diff)]
                    encoder_sum.reverse()

            encoder_values.append(encoder_sum)
        encoder_values = [j for i in encoder_values for j in i]
        encoder_values_Ninferences.append(encoder_values)
        
    compout_values_Ninferences = []
    for i in range(N_INFERENCES):
        compout_values = []
        for j in range(256):
            a = encoder_values_Ninferences[i][j]
            if(a==3):
                compout_values.append(7)
            elif(a==2):
                compout_values.append(3)
            elif(a==1):
                compout_values.append(1)
            else:
                compout_values.append(0)
        compout_values_Ninferences.append(compout_values)

    return compout_values_Ninferences