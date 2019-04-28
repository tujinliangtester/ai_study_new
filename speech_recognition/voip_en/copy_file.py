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


def copy_file(source_file, target_file):
    os.system('copy %s %s' % (source_file, target_file))


if __name__ == '__main__':
    # filepath='D:\\学习笔记\\ai\\dataSets\\data_voip_en\\tmpData'
    filepath = 'E:\\tjl_ai\\dataSet\\tmp\\'

    flist = get_file_names(filepath)
    source_file_path = 'E:\\tjl_ai\\dataSet\\data_voip_en\\data\\'
    print(flist[0])
    for file in flist:
        detail_name = file.split('\\')
        detail_name = detail_name[-1]
        source_file = source_file_path + detail_name + '.trn'
        target_file = filepath + detail_name + '.trn'
        print(source_file)
        copy_file(source_file, target_file)
    print('done')
