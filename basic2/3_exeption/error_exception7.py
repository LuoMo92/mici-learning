
# In[]
def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print("division by zero!")
    else:
        print("result is", result)
    finally:
        print("executing finally clause")

# In[]
divide(2, 1)
# In[]
divide(2, 0)
# In[]
divide("2", "1")


#%%
