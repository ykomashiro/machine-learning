def cal(num=3):
    num -= 1
    if num > 0:
        cal(num)
        return print(num)


if __name__ == "__main__":
    a = cal(10)
    print(a)
