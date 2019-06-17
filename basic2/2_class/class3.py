
# In[]:
# 类共享的变量与类的对象的变量
class Dog:
    # 类所共享的变量
    kind = 'canine'
    def __init__(self, name):
        # 每一个类的实例的变量
        self.name = name 

d = Dog('Fido')
e = Dog('Buddy')

# In[]:
print(d.kind) # 所有狗dog共享

print(e.kind) # shared by all dogs

print(d.name) # unique to d

print(e.name) # unique to e


#%%
