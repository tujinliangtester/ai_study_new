import os

tmp_list = []
def get_file_names(filepath):
    files = os.listdir(filepath)

    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_file_names(fi_d)
        else:
            tmp_list.append(fi_d)
            # print(os.path.join(filepath, fi_d))
    return tmp_list

def copy_file(source_file,target_file):
    os.system('copy %s %s' %(source_file,target_file))
if __name__=='__main__':
    filepath='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData'
    flist=get_file_names(filepath)
    print(flist[0])
    for file in flist:
        detail_name=file.split('\\')
        detail_name=detail_name[-1]
        source_file='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\data\\'+detail_name+'.TRN'
        target_file='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData\\'+detail_name+'.TRN'
        copy_file(source_file,target_file)
    print('done')