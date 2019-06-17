"""
函数：任意数量参数
"""
# In[]
def concat(*args, sep="/"):
    return sep.join(args)

print(concat("earth", "mars", "venus"))

print(concat("earth", "mars", "venus", sep="."))

# In[]:
print(concat("earth", "mars", "venus", "."))

# In[]
# 注意下面写法是有问题的，满足不了我们想要的一种方式与结果
def concat2(sep="/", *args):
    return sep.join(args)

# 这里会把earth作为sep参数
print(concat2("earth", "mars", "venus"))

# In[]:
# 这样写会出错，因为关键字参数必须方法非关键字参数的后面
# 必须修改为  print(concat2(".", "earth", "mars", "venus" ))
# 总体来说，这样的函数与我们需要的功能是有一定出入的
# 所以如果需要设置缺省的参数和连续多个变化参数
# 需要把关键字参数一定设置到*命名参数的后面

# In[]:
print(concat2(".", "earth", "mars", "venus" ))
# In[]:
print(concat2(sep=".", "earth", "mars", "venus" ))


#%%
