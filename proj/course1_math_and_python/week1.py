import numpy as np
import codecs
import utils


def header():
    return 'WEEK 1: Intro';


def run():
    print(np.__version__)
    basics()
    files()

    return


def basics():
    xs = [1, 2, 3, 4, 5]
    ys = [ y for y in map(lambda x: x**2, xs) ]
    print(ys)


def files():
    with open(utils.PATH. COURSE_FILE(1, 'test.txt'), 'r', encoding='utf8') as file:
        print(file.readline())
        print(file.read())

    with open(utils.PATH. COURSE_FILE(1, 'test.txt')) as file:
        for line in file:
           print(line.strip())

    with open(utils.PATH. COURSE_FILE(1, 'test-write.txt'), 'w') as file:
        file.write('hello writing')
