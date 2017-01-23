import sys
import os
import lib

if __name__ == '__main__':
    OPTIONS = lib.Options()
    OPTS = OPTIONS.parse(sys.argv[1:])

    print(sys.path)