def square(i):
    for i in range(i):
        if i>5:
            print(i)
            yield i

square(10)
for i in square(10):
    print(i)