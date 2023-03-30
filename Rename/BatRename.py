import os #
filename = 'C:/Users/jzhao/Documents/code/1data-lab/CSTQ/2Modify/Fluo-C2DL-MSC/02_Msk_CSTQ_Test' #file name
list_path = os.listdir(filename)  #read file name
for index in list_path:  #list_path
    name = index.split('.')[0]   #split,get first
    kid = index.split('.')[-1]   #get last
    path = filename + '/' + index
    new_path = filename +'/'+ name + '- jz how are you' + '.' + kid
    os.rename(path, new_path) # rename

print('Done')

