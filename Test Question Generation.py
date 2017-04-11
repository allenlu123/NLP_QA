import qa
import string

f = open('../Stuff/test1.txt')
t = f.read()

printable = set(string.printable)
t = filter(lambda x: x in printable, t)

print qa.generateQuestions(t)