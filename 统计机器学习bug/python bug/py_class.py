
#object 表示从object类继承
class Person(object):
    def __init__(self,name,age): # __init__的第一个参数永远都是self,self指向创建的实例本身,
        self.name = name;
        self.age = age
    def __str__(self):
        return '<Person: %s(%s)>' % (self.name, self.age)


if __name__ == '__main__':
    piglei = Person('piglei',24)
    print(piglei)