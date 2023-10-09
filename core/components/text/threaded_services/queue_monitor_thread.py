import logging
import threading
import time
import json
from datetime import timedelta

from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileCacheQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueueMonitorThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue):
		threading.Thread.__init__(self, name=name, daemon=True)
		self.file_stash = file_stash
		self.polling_interval: float = timedelta(minutes=1).total_seconds()

	def run(self):
		logger.info(":: Starting Queue-Monitor-Thread")
		self.monitor_stash()

	def monitor_stash(self):
		while True:
			try:
				types_to_check = [QueueType.POST, QueueType.GENERATION, QueueType.REPLY]
				statuses = []
				for queue_type in types_to_check:
					status: dict = self.file_stash.get_queue_status(queue_type)
					statuses.append(status)
				logger.info("Queue Monitor Thread - Queue Status")
				logger.info("-----------------------------------")
				logger.info(f"{'Queue Name':<20} | {'Queue Size'}")
				logger.info("-----------------------------------")
				for item in statuses:
					logger.info(f"{item['queue_name']:<20} | {item['queue_size']}")
			except Exception as e:
				logger.exception("Unexpected error: ", e)
			finally:
				time.sleep(self.polling_interval)
