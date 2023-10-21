import logging
import threading
import time
from datetime import timedelta, datetime

from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileQueue, FileCache

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueueMonitorThread(threading.Thread):
	def __init__(self, name: str, daemon: bool, file_queue: FileQueue, file_stash: FileCache):
		super().__init__(name=name, daemon=daemon)
		self.polling_interval: float = timedelta(minutes=1).total_seconds()
		self.file_stash: FileCache = file_stash
		self.file_queue: FileQueue = file_queue

	def run(self):
		logger.info(":: Starting Queue-Monitor-Thread")
		self.monitor_stash()


	def monitor_stash(self):
		while True:
			try:
				types_to_check = [QueueType.POST, QueueType.GENERATION, QueueType.REPLY]
				statuses = []
				for queue_type in types_to_check:
					status: dict = self.file_queue.get_queue_status(queue_type)
					statuses.append(status)
				current_time = datetime.now().timestamp()
				time_to_post: float = float(self.file_stash.cache_get("time_to_post"))
				reportable_time_in_seconds = time_to_post - current_time
				logger.info("Queue Monitor Thread - Queue Status - Time Till Next Post")
				logger.info("-----------------------------------")
				logger.info(f"{'Queue Name':<20} | {'Queue Size'} | {reportable_time_in_seconds}")
				logger.info("-----------------------------------")
				for item in statuses:
					if item is None:
						continue
					queue_name = item.get('queue_name')
					queue_size = item.get('queue_size')
					logger.info(f"{queue_name:<20} | {queue_size}")
			except Exception as e:
				logger.exception("Unexpected error: ", e)
			finally:
				time.sleep(self.polling_interval)
