from peewee_migrate import Router
from peewee import SqliteDatabase
import models


router = Router(models.db)

# Create migration
router.create('ladida')

# Run migration/migrations
router.run('ladida')

# Run all unapplied migrations
router.run()