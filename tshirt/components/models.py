from peewee import Model, SqliteDatabase, CharField, IntegerField
import datetime
import os


PS = os.path.dirname(os.path.abspath(__file__))
MYDB = SqliteDatabase(os.path.join(PS, '../../data/my_database.db'))

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
