import os
import csv
import numpy as np
import click

class HSDatalogConverter:

    @staticmethod
    #def to_cartesiam_format(destination_path, filename, pandas_df, n_samples, n_signals_desired = 60, n_files = 1):
    def to_cartesiam_format(destination_path, filename, pandas_df, n_samples, n_files = 1):    
        """"
        n_signals_desired: rows desired in the output dataset. It can be used 
                            to obtain balanced datasets
        n_files: number of ouput csv files. It is useful for separating training, 
                test and validation Dataset
                """

        # arrange data for cartesaim
        filtered_df = pandas_df.drop('Time',1)
        dataset = filtered_df.to_numpy()
        
        timestamps_per_file = np.shape(dataset)[0]/n_files

        if timestamps_per_file < n_samples:
            click.secho("Error: Not enough timestamps per file [{}]. Chosen n_samples value [{}] should be lower or equal to {}".format(timestamps_per_file, n_samples, timestamps_per_file), fg='red')
            return None

        n_signals = int(np.floor((np.shape(dataset)[0]/n_files)/n_samples)) #rearrange number of signals to the maximum number extractable from the array
        

        # #check if the number of signals desired is too much
        # if n_signals_desired <= n_signals:
        #     n_signals = n_signals_desired
        # else: 
        #     click.secho("Warning: the number of signals desired should be lower than {}, the maximum reachable with the current signal: number of signals set to {}".format(n_signals, n_signals), fg='yellow')
        
        signal = []
        idx = 0
        for ii in range(0, n_files):
            filename = filename + "_Cartesiam_segments_{}.csv".format(ii)
            
            file_path = os.path.join(destination_path,filename)
            with open(file_path , "w", newline="") as f:
                writer = csv.writer(f)
                for rr in range(0, n_signals): #rows of final dataset
                    for cc in range(idx, idx + n_samples): #cc = columns in input dataset
                        if cc >= dataset.shape[0]:
                            continue
                        for el in dataset[cc]:
                            signal.append(el)
        
                    idx += n_samples
                    writer.writerow(list(signal))
                    signal.clear()
        click.secho("--> File: \"" + filename + "\" correctly exported" , fg='green')

    @staticmethod
    def to_csv(df, filename):
        HSDatalogConverter.to_xsv(df, filename, '.csv', ',')

    @staticmethod
    def to_tsv(df, filename):
        HSDatalogConverter.to_xsv(df, filename, '.tsv', '\t')

    @staticmethod
    def to_xsv(df, filename, extension, separator):
        df.to_csv(filename + extension, separator)
        click.secho("--> File: \"" + filename+extension + "\" correctly exported" , fg='green')

    @staticmethod
    def to_unico():
        pass

    