import logging
import os
import shelve
from datetime import datetime, timedelta

from core.components.text.models.internal_types import QueueType

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FileCacheQueue:
	def __init__(self, db_name):
		self.db_name: str = db_name
		self.locked = False
		with shelve.open(self.db_name, writeback=True) as db:
			if 'time_to_post' not in db:
				db['time_to_post'] = (datetime.now() + timedelta(hours=3)).timestamp()

			if QueueType.GENERATION.value not in db:
				db[str(QueueType.GENERATION.value)] = []
			if QueueType.REPLY.value not in db:
				db[str(QueueType.REPLY.value)] = []
			if QueueType.POST.value not in db:
				db[str(QueueType.POST.value)] = []

	def cache_get(self, key):
		try:
			while self.locked:
				continue
			with shelve.open(self.db_name, writeback=True) as db:
				self.locked = True
				return db.get(key, None)
		except Exception as e:
			logger.exception(e)
		finally:
			self.locked = False

	def cache_set(self, key, value):
		try:
			while self.locked:
				continue
			try:
				with shelve.open(self.db_name, writeback=True) as db:
					self.locked = True
					db[key] = value
			except Exception as e:
				logger.exception(e)
		finally:
			self.locked = False

	def queue_put(self, value, queue_type: QueueType):
		try:
			while self.locked:
				continue
			with shelve.open(self.db_name, writeback=True) as db:
				self.locked = True
				queue = db[str(queue_type.value)]
				queue.append(value)
				db[queue_type.value] = queue
		except Exception as e:
			logger.exception(e)
		finally:
			self.locked = False

	def queue_pop(self, queue_name: QueueType):
		try:
			while self.locked:
				continue
			self.locked = True
			with shelve.open(self.db_name, writeback=True) as db:
				queue = db.get(str(queue_name.value), [])
				if not queue or len(queue) == 0:
					self.locked = False
					return None
				if len(queue) == 0:
					self.locked = False
					return None
				else:
					value = queue.pop()
					db[str(queue_name.value)] = queue
					self.locked = False
					return value
		except Exception as e:
			logger.exception(e)

	def get_queue_status(self, queue_name: QueueType):
		try:
			while self.locked:
				continue
			self.locked = True
			with shelve.open(self.db_name, writeback=True) as db:
				queue = db.get(str(queue_name.value), [])
				if not queue or len(queue) == 0:
					self.locked = False
					return {
						"queue_name": queue_name.value,
						"queue_size": 0,
					}
				if len(queue) == 0:
					self.locked = False
					return {
						"queue_name": queue_name.value,
						"queue_size": 0,
					}
				else:
					self.locked = False
					return {
						"queue_name": queue_name.value,
						"queue_size": len(queue),
					}
		except Exception as e:
			logger.exception(e)

