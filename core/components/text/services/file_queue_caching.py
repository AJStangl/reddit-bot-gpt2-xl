import logging
import shelve
import threading
from datetime import datetime, timedelta

from core.components.text.models.internal_types import QueueType

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileCacheQueue:
    def __init__(self, db_name):
        self.db_name: str = db_name
        self.lock = threading.Lock()
        with shelve.open(self.db_name, writeback=True) as db:
            if 'time_to_post' not in db:
                db['time_to_post'] = (datetime.now() + timedelta(hours=3)).timestamp()
            for queue_type in QueueType:
                if queue_type.value not in db:
                    db[queue_type.value] = []

    def cache_get(self, key):
        with self.lock:
            try:
                with shelve.open(self.db_name, writeback=True) as db:
                    return db.get(key, None)
            except Exception as e:
                logger.exception(e)

    def cache_set(self, key, value):
        with self.lock:
            try:
                with shelve.open(self.db_name, writeback=True) as db:
                    db[key] = value
            except Exception as e:
                logger.exception(e)

    def queue_put(self, value, queue_type: QueueType):
        with self.lock:
            try:
                with shelve.open(self.db_name, writeback=True) as db:
                    queue = db[queue_type.value]
                    queue.append(value)
                    db[queue_type.value] = queue
            except Exception as e:
                logger.exception(e)

    def queue_pop(self, queue_type: QueueType):
        with self.lock:
            try:
                with shelve.open(self.db_name, writeback=True) as db:
                    queue = db.get(queue_type.value, [])
                    if queue:
                        return queue.pop()
            except Exception as e:
                logger.exception(e)

    def get_queue_status(self, queue_type: QueueType):
        with self.lock:
            try:
                with shelve.open(self.db_name, writeback=True) as db:
                    queue = db.get(queue_type.value, [])
                    return {
                        "queue_name": queue_type.value,
                        "queue_size": len(queue),
                    }
            except Exception as e:
                logger.exception(e)