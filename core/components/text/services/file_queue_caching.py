import logging
import os
from typing import Optional

from filelock import FileLock

from core.components.text.models.internal_types import QueueType

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import json
import threading
from datetime import datetime, timedelta


class FileCache:
    def __init__(self, db_name: str, lock: threading.Lock):
        self.db_name = db_name + '.json'
        self.lock = lock
        with self.lock:
            try:
                with open(self.db_name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    db = json.loads(content)
            except FileNotFoundError:
                db = {}
            except Exception:
                db = {}

            if 'time_to_post' not in db:
                db['time_to_post'] = (datetime.now() + timedelta(hours=3)).timestamp()

            for queue_type in QueueType:
                if queue_type.value not in db:
                    db[queue_type.value] = []

            with open(self.db_name, 'w') as f:
                json.dump(db, f)

    def cache_get(self, key):
        with self.lock:
            try:
                with open(self.db_name, 'r') as f:
                    db = json.load(f)
                return db.get(key, None)
            except Exception as e:
                # Handle exception (replace logger with your logging mechanism)
                logger.exception(e)

    def cache_set(self, key, value):
        with self.lock:
            try:
                with open(self.db_name, 'r') as f:
                    db = json.load(f)
                db[key] = value
                with open(self.db_name, 'w') as f:
                    json.dump(db, f)
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
                    with open(queue_path, 'a+', encoding='utf-8') as f:
                        f.seek(0)
                        for line in f.readlines():
                            try:
                                if line.startswith('\x00'): # fuck if I know
                                    continue
                                self.queue[queue_name].append(json.loads(line))
                            except Exception as e:
                                logger.exception(e)
                                continue
        except Exception as e:
            logger.exception(e)

    def queue_put(self, value, queue_type: QueueType):
        try:
            queue_name = queue_type.value
            queue_path = os.path.join(self.queue_path, queue_name)
            with FileLock(queue_path + ".lock"), open(queue_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(value))
                f.write("\n")
                self.queue[queue_name].append(value)
        except Exception as e:
            logger.exception(e)

    def queue_pop(self, queue_type: QueueType):
        try:
            queue_name = queue_type.value
            queue_path = os.path.join(self.queue_path, queue_name)
            with open(queue_path, 'r', encoding='utf-8') as f:
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