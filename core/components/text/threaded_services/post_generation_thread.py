import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta

from core.components.text.models.internal_types import QueueType
from core.components.text.models.queue_message import RedditComment
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCache, FileQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostGenerationThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool, file_queue: FileQueue):
		super().__init__(name=name, daemon=daemon)
		self.file_stash: FileCache = file_stash
		self.next_time_to_post: float = self.initialize_time_to_post()
		self.config: ConfigurationManager = ConfigurationManager()
		self.file_queue: FileQueue = file_queue

	def run(self):
		logger.info(":: Starting Post-Generation-Thread")
		self.process_generation_queue()

	def initialize_time_to_post(self) -> float:
		next_post_time = self.file_stash.cache_get('time_to_post')
		if next_post_time is None:
			next_post_time = (datetime.now() + timedelta(hours=3)).timestamp()
			self.file_stash.cache_set('time_to_post', next_post_time)
			return next_post_time
		else:
			return next_post_time

	def process_generation_queue(self):
		while True:
			try:
				current_time = datetime.now().timestamp()
				if self.next_time_to_post > current_time:
					time.sleep(1)
					continue
				else:
					self.create_post_string_and_send_to_queue()
					self.next_time_to_post = float((datetime.now() + timedelta(hours=1)).timestamp())
					self.file_stash.cache_set('time_to_post', self.next_time_to_post)
			except Exception as e:
				logger.exception("Unexpected error: ", e)
				time.sleep(5)

	def create_post_string_and_send_to_queue(self) -> dict:
		topic = random.choice(self.config.read_topics_file())
		posting_bot = random.choice(list(self.config.bot_map.keys()))
		constructed_string = f"<|startoftext|><|subreddit|>r/{topic}<|title|>"
		data: dict = {
			'text': constructed_string,
			"image": "",
			"responding_bot": posting_bot,
			"subreddit": os.environ.get("SUBREDDIT_TO_MONITOR").split("+")[0],
			"reply_id": "",
			"title": "",
			"type": "post"
		}
		self.file_queue.queue_put(data, QueueType.GENERATION)
