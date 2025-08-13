import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from PyPDF2 import PdfMerger

# TO DO: generalize for different sensor geometries
#      - generalize offset values
#      - Change offset1/2 var names to offsetX/Y for better readability

def read_blocks_from_file(filename):
    with open(filename, 'r') as file:
        blocks = []
        block = []
        for line in file:
            if line.startswith('<time slice'):
                if block:
                    blocks.append(block)
                    block = []
            else:
                block.append([float(x) for x in line.split()])
        if block:
            blocks.append(block)
    print(blocks)
    return blocks

def apply_offset(block, offset, rows=13, cols=21):
    new_block = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if block[i][j] != 0:
                new_i = i + offset[0]
                new_j = j + offset[1]
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    new_block[new_i][new_j] = block[i][j]
    
    return new_block

def check_boundary(matrix, threshold=1):
    """
    Checks if there are non-zero elements at the boundary of the given matrix.
    """
    # Check the boundary elements
    top_row = matrix[0, :]  # Top row
    bottom_row = matrix[-1, :]  # Bottom row
    left_column = matrix[:, 0]  # Left column
    right_column = matrix[:, -1]  # Right column

    # If any boundary element is non-zero, set the flag to True
    if (np.any(np.abs(top_row) > threshold) or np.any(np.abs(bottom_row) > threshold) or np.any(np.abs(left_column) > threshold) or np.any(np.abs(right_column) > threshold)    ):
        return True
    else:
        return False

def split(index,df1,df2,df2_uncentered,df3,df3_uncentered):
        print("Columns of labels file: ", df1.columns)
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)
        df2_uncentered.columns = df2_uncentered.columns.astype(str)
        df3.columns = df3.columns.astype(str)
        df3_uncentered.columns = df3_uncentered.columns.astype(str)
        # unflipped, all charge              
        if("unflipped" not in os.listdir()):
            os.mkdir("unflipped")                                               
        df1[df1['z-entry']==100].to_parquet("unflipped/labels_d"+str(index)+".parquet")
        df2[df1['z-entry']==100].to_parquet("unflipped/recon2D_d"+str(index)+".parquet")
        df2_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon2D_uncentered_d"+str(index)+".parquet")
        df3[df1['z-entry']==100].to_parquet("unflipped/recon3D_d"+str(index)+".parquet")
        df3_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon3D_uncentered_d"+str(index)+".parquet")

def parseFile(filein,tag,timestep,nevents=-1,row_size=13,col_size=21):
        with open(filein) as f:
                lines = f.readlines()
        header = lines[0].strip()
        #header = lines.pop(0).strip()
        pixelstats = lines[1].strip()
        #pixelstats = lines.pop(0).strip()
        print("Header: ", header)
        print("Pixelstats: ", pixelstats)
        readyToGetTruth = False
        readyToGetTimeSlice = False
        clusterctr = 0
        cluster_truth =[]
        timeslice = 0
        cur_slice = []
        cur_cluster = []
        events = []
        if nevents == -1:
                nevents = math.inf
        for line in lines:
                ## Start of the cluster
                if "<cluster>" in line:
                        readyToGetTruth = True
                        readyToGetTimeSlice = False
                        clusterctr += 1
                        # Create an empty cluster
                        cur_cluster = []
                        timeslice = 0
                        # move to next line
                        continue
                # the line after cluster is the truth
                if readyToGetTruth:
                        cluster_truth.append(line.strip().split())
                        readyToGetTruth = False
                        # move to next line
                        continue
                ## Put cluster information into np array
                if "time slice" in line:
                        readyToGetTimeSlice = True
                        cur_slice = []
                        timeslice += 1
                        # move to next line
                        continue
                if readyToGetTimeSlice:
                        cur_row = line.strip().split()
                        cur_slice += [float(item) for item in cur_row]
                        # When you have all elements of the 2D image:
                        if len(cur_slice) == row_size*col_size:
                                cur_cluster.append(cur_slice)
                        # When you have all time slices:
                        if len(cur_cluster) == int(4000/timestep):
                                events.append(cur_cluster)
                                readyToGetTimeSlice = False
                if len(events) >= nevents:
                        break
        print("Number of clusters = ", len(cluster_truth))
        print("Number of events = ",len(events))
        print("Number of time slices in cluster = ", len(events[0]))
        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )
        return arr_events, arr_truth

def save_total_charge_data(arr_events, arr_truth, label, temp_plot_data, row_size=13, col_size=21, sensor_thickness=100):
        """
        Plots relaated to total charge vs time for an arbitrary event and histogram of total charge / path length across all events
        """
        # Initialize empty lists for labels and charges
        all_labels = []
        all_charges = []

        # Check if the temporary data file exists
        if os.path.exists(temp_plot_data):
                try:
                        # Load existing plot data
                        data = np.load(temp_plot_data, allow_pickle=True)
                        all_labels = list(data['labels']) if 'labels' in data else []
                        all_charges = list(data['charges']) if 'charges' in data else []
                except Exception as e:
                        print(f"Error loading {temp_plot_data}: {e}. Recreating the file.")
                        # If loading fails, reset the data
                        all_labels = []
                        all_charges = []

        # Avoid re-adding data for the same label
        if label not in all_labels:
                # Extract total charge for the first event
                first_event = arr_events[3]  # Shape: (time_slices, row_size * col_size)
                # Plotting total charge vs time for the first event
                total_charge_per_time_slice = [np.sum(time_slice) for time_slice in first_event]
                
                # # If you want to plot a specific pixel's output vs time, e.g., (6, 14):
                # pixel_row, pixel_col = 6, 14
                # total_charge_per_time_slice = [time_slice[pixel_row, pixel_col] for time_slice in first_event.reshape(-1, row_size, col_size)]

                # Append the new data
                all_labels.append(label)
                all_charges.append(total_charge_per_time_slice)
                np.savez(temp_plot_data, labels=np.array(all_labels, dtype=object), charges=np.array(all_charges, dtype=object))
        else:
                print(f"Data for label {label} already exists. Skipping saving.")

        # Calculate total charge across all events
        total_charges = []
        for event, truth in zip(arr_events, arr_truth):
                total_charge = np.sum(event.reshape(-1, row_size, col_size)[-1])#np.sum(event[-1])  # Sum all charges in the event
                angle_vector_mag = np.sqrt(truth[3]**2 + truth[4]**2 + truth[5]**2)
                path_len = sensor_thickness / (angle_vector_mag * np.abs(truth[5]))
                total_charges.append(total_charge/path_len)
        # Save total charges histogram as an .npz file
        np.savez(f'total_charges_hist_{label}ps_timestep.npz', total_charges=total_charges)

        # Plot histogram of total charges
        plt.figure(figsize=(8, 6))
        plt.hist(total_charges, bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title("Histogram of total charge/path length across all events")
        plt.xlabel("Total charge [e-]")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f'total_charge_histogram_{label}ps_timestep.pdf', format="pdf")
        plt.close()


def plot_total_charge_from_npz(temp_plot_data, output_pdf="total_charge_vs_time.pdf", total_time=4000):
        """
        Plot total charge vs time for an arbitrary event from the saved .npz file.
        """
        # Check if the temporary data file exists
        if not os.path.exists(temp_plot_data):
                print(f"No data file found: {temp_plot_data}. Cannot plot.")
                return

        try:
                # Load existing plot data
                data = np.load(temp_plot_data, allow_pickle=True)
                all_labels = list(data['labels']) if 'labels' in data else []
                all_charges = list(data['charges']) if 'charges' in data else []
        except Exception as e:
                print(f"Error loading {temp_plot_data}: {e}. Cannot plot.")
                return

        # Plot all curves
        plt.figure(figsize=(10, 6))
        for i, charges in enumerate(all_charges):
                time_steps = np.arange(all_labels[i], total_time + all_labels[i], all_labels[i])  # Uniform time axis for all curves
                plt.plot(time_steps, charges, marker='o', markersize=0.1, linestyle='-', label=f'{all_labels[i]} ps timestep')

        plt.legend()
        plt.title("Total charge vs time (for an arbitrary event)")
        # plt.title("Pixel charge vs time (for an arbitrary event)")
        plt.xlabel("Time [ps]")
        plt.ylabel("Total Charge [e-]")
        plt.grid(True)
        plt.savefig(output_pdf, format="pdf")
        plt.close()

def plot_histograms_from_npz(histogram_files, output_pdf="combined_histograms.pdf"):
    """
    Read multiple .npz histogram files (containing total charge / path length across all events) and plot them on the same plot.
    """
    if not histogram_files:
        print("No histogram files provided.")
        return

    plt.figure(figsize=(10, 6))

    for hist_file in histogram_files:
        if not os.path.exists(hist_file):
            print(f"File not found: {hist_file}. Skipping.")
            continue

        try:
            # Load histogram data from the .npz file
            data = np.load(hist_file, allow_pickle=True)
            total_charges = data['total_charges']

            # Extract label from the filename (assuming the label is part of the filename)
            label = os.path.basename(hist_file).split('_')[3]  # Adjust based on filename format

            # Plot the histogram as a line plot for comparison
            hist, bins = np.histogram(total_charges, bins=20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            plt.plot(bin_centers, hist, label=f"{label} ps timestep")
        except Exception as e:
            print(f"Error loading {hist_file}: {e}. Skipping.")
            continue

    # Add plot details
    plt.legend()
    plt.title("Histograms of total charge / path length")
    plt.xlabel("Total charge [e-]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_pdf, format="pdf")
    plt.close()

def main():
        sensor_pitch_X = 50 # in um
        sensor_pitch_Y = 12.5 # in um
        sensor_thickness = 100 #um      
        # row_size, col_size = 32, 32
        row_size, col_size = 13, 21
        print("==========================")                   
        print(f'NOTE - sensor geometry is hard-coded as {sensor_pitch_X}x{sensor_pitch_Y}x{sensor_thickness} um3. \nAssuming pixel array size = {col_size} X {row_size}/')
        print("==========================")                   
        index = int(sys.argv[1])
        timesteps = [200,50,10]
        tag = "d"+str(index)
        inputdir = "./"
        if os.path.exists("total_charge_vs_time.npz"):
                os.remove("total_charge_vs_time.npz")
                print(f"Removed existing file: \"total_charge_vs_time.npz\"")
        
        for timestep in timesteps[::-1]:
                if timestep == 200:
                        nevt = 5000
                else:
                        nevt = -1
                arr_events, arr_truth = parseFile(filein=inputdir+f'pixel_clusters_d{str(index)}_{timestep}ps.out', nevents=nevt, timestep=timestep, tag=tag, row_size=row_size, col_size=col_size)
                arr_truth = np.array(arr_truth, dtype=float)  # Convert all elements to float
                save_total_charge_data(arr_events, arr_truth, label=timestep, temp_plot_data= "total_charge_vs_time.npz", row_size=row_size, col_size=col_size, sensor_thickness=sensor_thickness)
        plot_total_charge_from_npz("total_charge_vs_time.npz", output_pdf="total_charge_vs_time.pdf", total_time=4000)
        histogram_files = [f"total_charges_hist_{timestep}ps_timestep.npz" for timestep in timesteps][::-1]
        plot_histograms_from_npz(histogram_files, output_pdf="combined_histograms.pdf")
        sys.exit()
        
if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

