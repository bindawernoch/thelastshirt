from peewee import Model, SqliteDatabase, CharField, IntegerField
import datetime
import pathlib
import socket
import os


CWD = pathlib.Path().cwd()
FILE_PATH = pathlib.Path(__file__).parent.absolute()
MY_HOME = pathlib.Path().home()

if socket.gethostname() == 'penguin':
    DB_FILE = MY_HOME / "./data/my_database.db"
else:
    DB_FILE = FILE_PATH / "../data/my_database.db"

MYDB = SqliteDatabase(str(DB_FILE), 
                                                pragmas={
                                                                    'journal_mode': 'TRUNCATE',
                                                                    # 'locking_mode': 'EXCLUSIVE',
                                                                    'cache_size': -1 * 64000,  # 64MB
                                                                    'foreign_keys': 1,
                                                                    'ignore_check_constraints': 0,
                                                                    'synchronous': 0} , )

class Classifications(Model):
    name = CharField()
    xstart = IntegerField()
    xend = IntegerField()
    ystart = IntegerField()
    yend = IntegerField()

    class Meta:
        database = MYDB

def create_my_tables():
    MYDB.connect()
    MYDB.create_tables([Classifications])

def get_classifications():
    query = Classifications.select()
    return list(query.dicts())
