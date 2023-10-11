import logging
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

	def run_generator(self):
		logger.info(":: Starting Generative Services")
		self.generative_services.create_prompt_completion(self.warp_up_prompt)
		while True:
			try:
				data_thing: Optional[dict] = self.file_queue.queue_pop(QueueType.GENERATION)
				if data_thing is None:
					time.sleep(1)
					continue
				else:
					data_thing_prompt = data_thing.get('text')
					data_thing_type = data_thing.get('type')
					result = None
					if data_thing_type == 'submission':
						result: str = self.generative_services.create_prompt_completion(data_thing_prompt)
					if data_thing_type == 'comment':
						result: str = self.generative_services.create_prompt_completion(data_thing_prompt)
					if data_thing_type == 'post':
						result: dict = self.generative_services.create_prompt_for_submission(data_thing_prompt)
						if result is None:
							continue
						data_thing['text'] = result.get('text')
						data_thing['image'] = result.get('image')
						data_thing['title'] = result.get('title')
						self.file_queue.queue_put(result, QueueType.POST)
						continue
					if result is None:
						continue
					else:
						data_thing['text'] = result
						self.file_queue.queue_put(data_thing, QueueType.REPLY)
			except Exception as e:
				logger.exception("Unexpected error: ", e)
				continue
