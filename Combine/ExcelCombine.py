 # importing the required modules
import glob
import pandas as pd

# specifying the path to csv files
# specifying the path to csv files
#path = "C:/Users/jzhao/Documents/code/1data-lab/CSTQ/2Modify/Fluo-C2DL-Huh7/Auto-Combine/forCombine"

path ="C:/Users/jzhao/Documents/OneDrive - Delaware State University/学习/3DSU-PhD/02-Research/02Math-03CSTQ/06Result/1CSTQ-PCA/2modify/MSC02-03"

# csv files in the path
file_list = glob.glob(path + "/*.xlsx")

# list of excel files we want to merge.
# pd.read_excel(file_path) reads the
# excel data into pandas dataframe.
excl_list = []

for file in file_list:
	excl_list.append(pd.read_excel(file))

# concatenate all DataFrames in the list
# into a single DataFrame, returns new
# DataFrame.
excl_merged = pd.concat(excl_list, ignore_index=True)

# exports the dataframe into excel file
# with specified name.
excl_merged.to_excel('Combine-result.xlsx', index=False)
