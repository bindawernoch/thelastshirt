import sys
import os
import lib

if __name__ == '__main__':
    OPTIONS = lib.Options()
    OPTS = OPTIONS.parse(sys.argv[1:])

    lib.run_thresholds("/home/mario/Dropbox/Tshirts/tshirt_proj/data/")
