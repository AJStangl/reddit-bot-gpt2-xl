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
			with shelve.open(self.db_name) as db:
				self.locked = True
				queue = db[str(queue_name.value)]
				if not queue:
					return None
				value = queue.pop(0)
				db[str(queue_name.value)] = queue
				return value
		except Exception as e:
			logger.exception(e)
		finally:
			self.locked = False
