
# In[]:
# 属性与方法的基本定义
class See:
    time = '20101103'
    place = 'tnn'
    def jump(self):
        return 'I like jumping'
    def eat(self):
        return 'I like banana'
    def sleep(self):
        return 'sleep is so annoying'

print('Class see is created')

# In[]:
see1 = See()
print(see1.time)
print(see1.jump())


# In[]:
# 类的构造函数
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)
print(x.r, x.i)

# In[]:
y = Complex(4, 89)
print(y.r, y.i)
# In[]:
# 在类的对象中动态增加一个属性
x.counter = 1

# In[]:
while x.counter < 10:
    x.counter = x.counter * 2

print(x.counter)

# In[]:
# 删除动态增加的属性
del x.counter


#%%
