import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta

from core.components.text.models.internal_types import QueueType
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCache, FileQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostGenerationThread(threading.Thread):
	def __init__(self, name: str, daemon: bool, file_stash: FileCache,  file_queue: FileQueue):
		super().__init__(name=name, daemon=daemon)
		self.file_stash: FileCache = file_stash
		self.next_time_to_post: float = self.initialize_time_to_post()
		self.config: ConfigurationManager = ConfigurationManager()
		self.file_queue: FileQueue = file_queue
		self.time_to_sleep_for_new_post: int = int(os.environ.get("HOURS_BETWEEN_POST"))
		self.topics_list = open(os.environ.get("TOPICS_PATH"), 'r', encoding='utf-8').read().splitlines()

	def run(self):
		logger.info(":: Starting Post-Generation-Thread")
		self.process_generation_queue()

	def initialize_time_to_post(self) -> float:
		next_post_time = self.file_stash.cache_get('time_to_post')
		if next_post_time is None:
			next_post_time = (datetime.now() + timedelta(hours=self.time_to_sleep_for_new_post)).timestamp()
			self.file_stash.cache_set('time_to_post', next_post_time)
			return next_post_time
		else:
			return next_post_time

	def process_generation_queue(self) -> None:
		while True:
			try:
				current_time = datetime.now().timestamp()
				if self.next_time_to_post > current_time:
					time.sleep(60)
					continue
				else:
					self.create_post_string_and_send_to_queue()
					self.next_time_to_post = float((datetime.now() + timedelta(hours=self.time_to_sleep_for_new_post)).timestamp())
					self.file_stash.cache_set('time_to_post', self.next_time_to_post)
					time.sleep(60)
			except Exception as e:
				logger.exception(e)
				time.sleep(5)

	def create_post_string_and_send_to_queue(self) -> dict:
		posting_bot = random.choice(list(self.config.bot_map.keys()))
		random_topic = random.choice(self.topics_list)
		constructed_string = f"<|startoftext|><|subreddit|>r/{random_topic}<|title|>"
		data: dict = {
			'text': constructed_string,
			"image": "",
			"responding_bot": posting_bot,
			"subreddit": next(os.environ.get("SUBREDDIT_TO_MONITOR").split("+").__iter__()),
			"reply_id": "",
			"title": "",
			"type": "post"
		}
		self.file_queue.queue_put(data, QueueType.GENERATION)
