"""
在类的外面定义函数
"""
# In[]:
def f1(self, x, y):
    return min(x, x+y)

class C:
    f = f1

    def func(self):
        return 'hihi'

    def g(self):
        return 'hello world'

    h = g

c = C()
print(c.f(1,2))

# In[]:
print(c.g())
print(c.h())

# In[]:
# 类函数的互相调用
class Bag:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def addtwice(self, x):
        self.add(x)
        self.add(x)

one_bag = Bag()
one_bag.add(1)
print(one_bag.data)

# In[]:
one_bag.addtwice(2)
print(one_bag.data)

#%%
