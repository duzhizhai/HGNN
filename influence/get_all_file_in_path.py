import os


def traverse_dir(current_dir, deep=0):
    """
    current_dir: 绝对路径: "./"  或当前路径: 'D:/picture'
    """
    dir_list = os.listdir(current_dir)
    # traverse folder first.
    path_list, file_list = [], []
    for dir in dir_list:
        path = os.path.join(current_dir, dir)
        if os.path.isdir(path):
            path_list.append(dir)
        else:
            file_list.append(dir)
    dir_list = path_list + file_list

    # traverse all dir.
    for dir in dir_list:
        path = os.path.join(current_dir, dir)
        if os.path.isdir(path):
            # do something to this directory
            print("\t" * deep, dir)
            traverse_dir(path, deep + 1)
        if os.path.isfile(path):
            # do something to this file
            print("\t" * deep, "|--", dir)


traverse_dir("./")
