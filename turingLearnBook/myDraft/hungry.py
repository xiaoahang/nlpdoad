# encoding: utf-8
print("hello world")
type(10)

# 数组
a = [1, 2, 3, 4, 5]
print(a)
len(a)

# 切片
a[0:2]
a[1:]
a[:3]
a[:-1]  # index -1 is the symbol of the last element
a[:-2]  # index -2 means the last 2 element

# 字典 （对象）
me = {'height': 180}  # 生成字典
me['height']  # 访问元素
me['weight'] = 70  # 添加新元素
me['weight']  # 访问元素
print(me)  # {'height': 180, 'weight': 70}

# 布尔型 针对bool型的运算符有3种 and、or 和 not
hungry = True
sleepy = False
type(hungry)
not hungry
hungry and sleepy
hungry or sleepy

# if语句
if hungry:
    print('I am hungry')
elif sleepy:
    print('I am sleepy')
else:
    print('what???')

    # py 的语法是使用缩紧 分离分支

# 循环语句 按顺序访问集合中的各个元素
for i in [1, 2, 3, 0]:
    print(i)


# 函数
def hello(a):
    print('hello ' + a + ' ~')


hello('Cerssi')


# 关闭py解释器 mac：ctrl+d d      win：ctrl+z enter
# 运行python脚本 ： 进入到这个文件的目录 然后输入命令 ： python hungry.py

# 类
class Man:
    def __init__(self, name):
        self.name = name
        print('Initialized!!')

    def hello(self):
        print('Hello ' + self.name + ' !')

    def goodbye(self):
        print('goodbye' + self.name + ' !')


m = Man("David")
m.hello()
m.goodbye()

# 这里我们定义了一个新类 Man。上面的栗子中，类Man生成了实例（对象）m。
# 类Man的构造函数（初始化方法）会接受参数name，然后用这个参数初始化实例变量self.name。实例变量是存储在各个实例中的变量。
# Python中可以想self.name这样，通过在self后面添加属性名来生成或访问实例变量
