import logging
import os
import threading
import time
from typing import Optional

from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileQueue
from core.components.text.services.text_generation import GenerativeServices

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextGenerationThread(threading.Thread):
	def __init__(self, name: str, daemon: bool, generative_services: GenerativeServices, file_queue: FileQueue):
		self.generative_services: GenerativeServices = generative_services
		self.file_queue: FileQueue = file_queue
		self.warp_up_prompt: str = "<|startoftext|><|subreddit|>r/things<|title|>What is your favorite color?<|text|>We are the knights who say ni!<|context_level|>0<|comment|>"
		super().__init__(name=name, daemon=daemon)

	def run(self):
		logging.info(":: Starting Text-Generation-Thread")
		self.run_generator()

	def handle_post_generation(self, prompt: str, data: dict) -> dict:
		result: dict = self.generative_services.create_prompt_for_submission(prompt=prompt)
		if result is None:
			return data
		data['text'] = result.get('text')
		data['image'] = result.get('image')
		data['title'] = result.get('title')
		logger.info(":: Sending data_thing to post queue")
		self.file_queue.queue_put(data, QueueType.POST)
		return None


	def handle_reply_generation(self, prompt: str, data: dict):
		result: str = self.generative_services.create_prompt_completion(prompt=prompt)
		data['text'] = result
		self.file_queue.queue_put(data, QueueType.REPLY)
		return None

	def run_generator(self):
		self.generative_services.create_prompt_completion(self.warp_up_prompt)
		while True:
			try:
				image_lock_path = os.path.join(os.environ.get("LOCK_PATH"), "sd.lock")
				if os.path.exists(image_lock_path):
					time.sleep(1)
					continue
				data_thing: Optional[dict] = self.file_queue.queue_pop(QueueType.GENERATION)
				if data_thing is None:
					continue
				else:
					data_thing_prompt = data_thing.get('text')
					data_thing_type = data_thing.get('type')
					if data_thing_type == 'submission' or data_thing_type == 'comment':
						self.handle_reply_generation(prompt=data_thing_prompt, data=data_thing)
						continue
					if data_thing_type == 'post':
						self.handle_post_generation(prompt=data_thing_prompt, data=data_thing)
						continue

			except Exception as e:
				logger.exception("Unexpected error: ", e)
				continue


