"""
最终都会发生的语句，不管是否有异常发生
一般用于资源的释放和清理
"""

# In[]
try:
    # 将is_error改成False和True进行测试
    is_error  = True
    if is_error:
        raise NameError('异常出现')
    else:
        print('no error')

finally:
    print('再见了,不管怎样，我都要执行!')

#%%
