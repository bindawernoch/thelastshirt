from peewee_migrate import Router
from peewee import SqliteDatabase
# tshirt
import models


models.create_my_tables()  # boot up my model as defined in models
#
router = Router(models.MYDB)
# Create migration
router.create('ladida')
# Run migration/migrations
router.run('ladida')
# Run all unapplied migrations
router.run()