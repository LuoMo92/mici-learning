"""
变量作用域
"""
# In[]:
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"

    # 只是作用于上面一层的 scope_test 范围内，所以spam会显示
    # scope_test范围最近的spam变量值，如果没有就显示函数中缺省值
    do_local()
    print("After local assignment:", spam)

    # 在下面函数中，noncloal关键字将spam的范围只是设置到
    # 本函数的范围内，所以尽管函数前面有一个spam值，也不会
    # 对函数中的spam值产生影响，这里所谓的local指的是函数
    # 所在的当前环境，这里就是scope_test函数
    do_nonlocal()
    print("After nonlocal assignment:", spam)

    # 函数中把spam设置为全局变量，全局变量是为了在全局函数和类中
    # 传递参数中，所以只有函数返回的时候才会发生变化
    # 尽管这里设置了spam全局变量，也给了值，但是scope_test范围内
    # 依然是以前的值
    do_global()
    print("After global assignment:", spam)
 
scope_test()
# 由于spam已经被设置为global属性，作用域为整个文件
# 此时将scope_test函数中的全局值进行返回。
# gobal的作用相当于传递参数，在函数外声明的变量如果想在函数内变化
# 就需要gobal声明，下面再看一个例子
print("In global scope:", spam)

# In[]:
count = 1
def do1():
    num = count
    return num

do1()

# In[]:
# 下面程序会报错，因为函数do想改变函数外的变量
count = 1
def do2():
    count  = count + 1
    return count

do2()

# In[]:
count = 1
def do3():
    global count
    count = count + 1
    return count
# 运行一次输出2
do3()
# In[]:
# 再运行一次,结果是3
do3()
# In[]:

#%%
