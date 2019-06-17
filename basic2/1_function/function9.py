"""
传入小函数作为参数
"""

# In[1]:
# 对list排序
aList = ['123', 'xyz', 'zara', 'abc', 'xyz']
aList.sort()
print(aList)

# In[2]:
# 还是list排序，list的元素是有两个元素的turple，缺省按照首个元素即索引0位置的元素排序 
pairs = [(1, 'one'), (3, 'three'),  (2, 'two'), (4, 'four')]
pairs.sort()
print(pairs)

# In[3]:
# 表示按照list中turple元素的第一个索引元素进行排序
# 对于匿名函数来说，输入参数pair指的就是list中的每个元素
# pair[1]表示取这个元素的第一个位置元素进行sort的排序关键字
pairs.sort(key=lambda pair: pair[1])
print(pairs)

#%%
