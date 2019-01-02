from peewee_migrate import Router
from peewee import SqliteDatabase
import sys
import os
# tshirt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components import models
import models


def main():
    models.create_my_tables()  # boot up my model as defined in models
    #
    router = Router(models.MYDB)
    # Create migration
    router.create('ladida')
    # Run migration/migrations
    router.run('ladida')
    # Run all unapplied migrations
    router.run()

if __name__ == '__main__':
    main()
