"""
类的继承
"""

# In[]:
class animal():
    def __init__(self):
        self.height=10
        self.weight=50

    def deception(self):
        print("ansible.height:"+str(self.height)+" animal.weight:"+str(self.weight))

    def run(self):
        print("animal is running....")

class dog(animal):
    def __init__(self):
       self.weight = 30
       self.height = 40

d = dog()
d.deception()
d.run()

print('\n')


# In[]:
class Base1:
    def __init__(self):
        print('This is Base init function')

# In[]:
class A1(Base1):
    def __init__(self):
        Base1.__init__(self)
        print('This is init function of A')

A1()

# In[]:
class B1(Base1):
    def __init__(self):
        Base1.__init__(self)
        print('This is init function of B')
B1()

# In[]:
# 这里是多重继承，Base1的Init会初始化两次，这显然是不对的。
# 所以这里初始化父类对象的方法是有问题的
class C1(A1,B1):
    def __init__(self):
        A1.__init__(self)
        B1.__init__(self)
        print('This is init function of C')

C1()

# In[]:
class Base:
    def __init__(self):
        print('This is Base init function')
        
class A(Base):
    def __init__(self):
        super().__init__()
        print('This is init function of A')
A()

# In[]:
class B(Base):
    def __init__(self):
        super().__init__()
        print('This is init function of B')
B()

# In[]:
class C(A,B):
    def __init__(self):
        super().__init__()
        print('This is init function of C')

C()

# In[]:
# pass本身没有任何意义，为了保持程序的完整性
class Animal(object):
  pass

class Mammal(Animal):
  pass
 
class Bird(Animal):
  pass

class Runnable(object):
  def run(self):
    print('Running...')
 
class Flyable(object):
  def fly(self):
    print('Flying...')

class Dog(Mammal, Runnable):
  pass

d = Dog()
d.run()

