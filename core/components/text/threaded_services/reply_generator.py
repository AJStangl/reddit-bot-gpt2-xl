import logging
import threading
import time
from typing import Optional

from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileCacheQueue
from core.components.text.services.text_generation import GenerativeServices

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextGenerationThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue):
		threading.Thread.__init__(self, name=name)
		self.generative_services: GenerativeServices = GenerativeServices()
		self.file_stash = file_stash
		self._stop_event = threading.Event()

	def run(self):
		logging.info("Starting Text-Generation-Thread")
		self.run_generator()

	def run_generator(self):
		"""
			{
			  'text': "<|startoftext|><|subreddit|><|title|><|text|><|context_level|>0<|comment|>",
			  'responding_bot': 'MeganBotGPT',
			  'subreddit': 'r/TheRatEmpire',
			  'reply_id': 'k3y4kwn',
			  'type': 'comment, submission, post'
			}
		"""
		while True:
			try:
				data_thing: Optional[dict] = self.file_stash.queue_pop(QueueType.GENERATION)
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
						data_thing['text'] = result.get('text')
						data_thing['image'] = result.get('image')
						data_thing['title'] = result.get('title')
						self.file_stash.queue_put(result, QueueType.POST)
						continue
					if result is None:
						continue
					else:
						data_thing['text'] = result
						self.file_stash.queue_put(data_thing, QueueType.REPLY)
			except Exception as e:
				logger.exception("Unexpected error: ", e)
				time.sleep(5)