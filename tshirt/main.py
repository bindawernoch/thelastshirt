import sys
import os
from lib import Options, Project

if __name__ == '__main__':
    options = Options()
    opts = options.parse(sys.argv[1:])

    v = Project(opts)

    v.date()
    v.print_example_arg()