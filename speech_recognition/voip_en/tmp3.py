import os

get_file_names_tmp_list = []
def get_file_names(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_file_names(fi_d)
        else:
            get_file_names_tmp_list.append(fi_d)
    return get_file_names_tmp_list

if __name__=='__main__':
    fpath = 'D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\'
    whole_file_list = get_file_names(fpath)
    wav_file_list = []
    trn_file_list = []
    for file in whole_file_list:
        if (file.find('TRN') >= 0):
            trn_file_list.append(file)
        else:
            wav_file_list.append(file)
    num_list=[]
    for file in trn_file_list:
        with open(file) as f:
            res=f.read()
            res=res.split(' ')
            num_list.append(len(res))
    print(max(num_list))