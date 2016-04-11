import sys
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs

[a, b, c, d, e] = DefaultArgs(sys.argv[1:], [11, 22, 33, 44, 55])

print a, b, c, d, e
