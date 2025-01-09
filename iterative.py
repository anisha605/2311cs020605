def add(x,y):
    return x+y
res = add(10,20)
print(res)


result = lambda x,y:x+y
print(result(10,20))

result = lambda x,y:x*y
print(result(10,20))
##iterable
list = [1,2,3,4,5,6,7,8,9,10]
for i in list:
    print(i)

##iteraror
list = [1,2,3,4,5,6,7,8,9,10]
iterate_var=iter(list)
type(iterate_var)
for i in iterate_var:
    if i % 2 == 0:
        print(i)