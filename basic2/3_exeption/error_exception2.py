import sys

try:
    data_dir = 'E:/pycharm-workspace/mici-learning/basic2/3_exeption/'
    # 把myfile.txt修改为myfile1.txt再测试
    f = open(data_dir + 'myfile1.txt')
    s = f.readline()
    i = int(s.strip())
except OSError as err:
    print("OS error: {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.", sys.exc_info()[0])
    # 分别测试下面两种情况，raise屏蔽后，打开下面的print
    # raise
    print('test')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise 