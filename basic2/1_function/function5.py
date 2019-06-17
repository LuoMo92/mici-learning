"""
函数：当最后一个参数是**命名，可以接受key-value的输入
*命名可以接受变化的多个参数，如果**存在，必须在**的前面
"""

# In[]:
def play(title,title2, *dialogues, **actors):
    print("-- Will you watch", title, "today?")
    print("-- I am not familiar with", title2)
    print("-- It is the one where they say dialogues like:")
    for dial in dialogues:
        print("&&", dial)
    print("-- Who will be performing?")
    for act in actors:
        print(act, ":", actors[act])

# In[]:
play("Romeo and Juliet",
     "Romeo and Juliet2",
     "Oh, Romeo, Romeo.",
     "I am here Juliet!",
     "IIIIII!",
     Romeo="Leonardo di Caprio", 
     Juliet="Eva Green",
     Juliet2="Eva Green2")

# In[]:
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])

# In[]:
cheeseshop("Limburger", 
            "It's very runny, sir.",
            "It's really very, VERY runny, sir.",
            shopkeeper="Michael Palin",
            client="John Cleese",
            sketch="Cheese Shop Sketch")



# In[]:
cheeseshop("Limburger", 
            "It's very runny, sir.",
            "It's really very, VERY runny, sir.",
            "ok,ok",
            shopkeeper="Michael Palin",
            client="John Cleese",
            sketch="Cheese Shop Sketch",
            another="hihi")


#%%
