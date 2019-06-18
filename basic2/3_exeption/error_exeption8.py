
# In[]
data_dir = 'D:/lyp/2017_2018_2/python_code/MTrain/MachineLearn/3_ML/0_python_basic/basic2/3_exeption/'
for line in open(data_dir + "myfile.txt"):
    print(line, end="")

# In[]
"""
上面这段代码的问题在于在代码执行完后没有立即关闭打开的文件。
这在简单的脚本里没什么，但是大型应用程序就会出问题。
with 语句使得文件之类的对象可以 确保总能及时准确地进行清理

语句执行后，文件 f 总会被关闭，即使是在处理文件中的数据时出错也一样。
其它对象是否提供了预定义的清理行为要查看它们的文档
"""

# In[]
with open(data_dir+"myfile.txt") as f:
    for line in f:
        print(line, end="")



#%%
