train.ipynb -> Adapted from GDG's notebook. Used to process data, train the algorithm, and save the model file + csv files. The capability to add gaussian noise to the data also has been added (can be utilized using the `inject_noise` bool).
prepare_weights.py -> Uses the output csv files (b5,c5,b2,c2) from above notebook and produces the final csv file (b5_w5_b2_w2_pixel_bin) that will be sent to the ASIC.

run.ipynb -> produces compouts needed for ASIC DNN data-taking.
efficiency.ipynb -> analyzes DNN + model outputs and produces performance plots/results.


[`Depreciated!`] pre_processing.ipynb -> Processes parqueted simulation datasets to quantized versions and produces csv files needed for training and evaluating the NN algorithm.

[`Depreciated!`] SimpleNN_Qkeras_DNN.ipynb -> train algorithm on processed data and save model file + csv file to program weights and biases on chip.