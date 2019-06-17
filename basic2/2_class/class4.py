"""
关于list对象作为类的成员变量/属性的一个例子

"""

# In[]:
class Dog:
    # tricks是每个狗独有的，所以作为类的共享变量不合适
    tricks = [] 

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

# In[]:
d = Dog('Fido')
e = Dog('Buddy')

d.add_trick('roll over')
e.add_trick('play dead')

# In[]:
# 不希望被共享
print(d.tricks)
print(e.tricks)

# In[]:
class Dog2:
    def __init__(self, name):
        self.name = name
         # 给每个狗创建单独的属性变量
        self.tricks = []

    def add_trick(self, trick):
        self.tricks.append(trick)

d = Dog2('Fido')
e = Dog2('Buddy')
d.add_trick('roll over')
e.add_trick('play dead')
print(d.tricks)
print(e.tricks)
