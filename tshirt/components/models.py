from peewee import *
import datetime
import os


PS = os.path.dirname(os.path.abspath(__file__))

db = SqliteDatabase(os.path.join(PS, '../../data/my_database.db'))

class Classifications(Model):
    name = CharField()
    xstart = IntegerField()
    xend = IntegerField()
    ystart = IntegerField()
    yend = IntegerField()

    class Meta:
        database = db # This model uses the "people.db" database.

db.connect()
db.create_tables([Classifications])