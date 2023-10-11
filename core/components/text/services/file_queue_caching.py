import json
import logging
import multiprocessing
import os
import shelve
import threading
from datetime import datetime, timedelta

from filelock import FileLock

from core.components.text.models.internal_types import QueueType

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileCache:
    def __init__(self, db_name, lock: threading.Lock):
        self.db_name: str = db_name
        self.lock: threading.Lock = lock
        with self.lock:
            with shelve.open(self.db_name, writeback=True) as db:
                try:
                    if 'time_to_post' not in db:
                        db['time_to_post'] = (datetime.now() + timedelta(hours=3)).timestamp()
                    for queue_type in QueueType:
                        if queue_type.value not in db:
                            db[queue_type.value] = []
                except Exception as e:
                    logger.exception(e)

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


class FileQueue:
    def __init__(self):
        self.queue_path = os.environ.get("QUEUE_FILE_PATH")
        self.queue = {}
        self.initialize_queue()
        self.lock = threading.Lock()

    def initialize_queue(self):
        try:
            for queue in QueueType:
                queue_name = queue.value
                queue_path = os.path.join(self.queue_path, queue_name)
                os.makedirs(os.path.dirname(queue_path), exist_ok=True)
                self.queue[queue_name] = []
                with FileLock(queue_path + ".lock"):
                    with open(queue_path, 'a+') as f:
                        f.seek(0)
                        for line in f.readlines():
                            self.queue[queue_name].append(json.loads(line))
        except Exception as e:
            logger.exception(e)

    def queue_put(self, value, queue_type: QueueType):
        try:
            queue_name = queue_type.value
            queue_path = os.path.join(self.queue_path, queue_name)
            with FileLock(queue_path + ".lock"), open(queue_path, 'a') as f:
                f.write(json.dumps(value))
                f.write("\n")
                self.queue[queue_name].append(value)
        except Exception as e:
            logger.exception(e)

    def queue_pop(self, queue_type: QueueType):
        try:
            queue_name = queue_type.value
            queue_path = os.path.join(self.queue_path, queue_name)
            with open(queue_path, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                 return None
            with open(queue_path, 'w') as f:
                for line in lines[1:]:
                    f.write(line)
            return json.loads(lines[0])
        except Exception as e:
            logger.exception(e)
            return None

    def get_queue_status(self, queue_type: QueueType):
        queue_name = queue_type.value
        queue_path = os.path.join(self.queue_path, queue_name)
        try:
            with open(queue_path, 'r') as f:
                lines = f.readlines()
                return {
                    "queue_name": queue_type.value,
                    "queue_size": len(lines),
                }
        except Exception as e:
            logger.exception(e)