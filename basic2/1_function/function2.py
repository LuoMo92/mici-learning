"""
函数缺省值：按照前面最近变量值设置参数缺省值
"""

# 函数按照前面变量的值设置缺失值
i = 5
def f(arg=i):
    print(arg)

i = 6
f()
f(100)