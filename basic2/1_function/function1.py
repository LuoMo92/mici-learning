"""
函数的缺省值
为了接受收入，使用外部终端方式external terminal运行
"""

def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False    
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


while(True):
    num = int(input('Input test num:'))
    if num == 1:
        ask_ok("Do you really want to quit?")
        print("Test over 1")
    elif num == 2:
        ask_ok("To overwrite?", 2)
        print("Test over 2")
    elif num == 3:
        ask_ok("To overwrite?", 2, "Please input y,ye,yes, n,no...")
        print("Test over 3")
    else:
        print('You quit!')
        break

