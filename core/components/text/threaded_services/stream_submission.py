import logging
import os
import random
import threading
import time

import praw
import prawcore
from praw.models import Submission

from core.components.text.models.internal_types import QueueType
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCache, FileQueue
from core.components.text.services.text_generation import ImageCaptioning

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubmissionHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool, file_queue: FileQueue, reddit: praw.Reddit):
		super().__init__(name=name, daemon=daemon)
		self.reddit: praw.Reddit = reddit
		self.sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		self.file_stash: FileCache = file_stash
		self.file_queue: FileQueue = file_queue
		self.config = ConfigurationManager()
		self.image_captioning: ImageCaptioning = ImageCaptioning()

	def run(self):
		logger.info(":: Starting Submission-Handler-Thread")
		self.process_subreddit_stream()

	def process_subreddit_stream(self):
		subreddit = self.reddit.subreddit(self.sub_names)
		while True:
			try:
				self.process_submissions_in_stream(subreddit)
			except prawcore.exceptions.TooManyRequests as e:
				logger.exception(e)
				time.sleep(10)
				continue
			except Exception as e:
				logger.exception(e)
				time.sleep(10)
				continue


	def process_submissions_in_stream(self, subreddit):
		for item in subreddit.stream.submissions(pause_after=0, skip_existing=True):
			if item is None:
				time.sleep(30)
				continue

			if isinstance(item, Submission):
				self.process_submission(submission=item)


	def process_submission(self, submission) -> None:
		if submission is None:
			time.sleep(5)
			return None

		if int(submission.num_comments) >= int(os.environ.get("MAX_REPLIES")):
			return None

		bots_to_reply = list(self.config.bot_map.keys())
		for bot in self.config.bot_map.keys():
			bot_reply_key = f"{bot}-{submission.id}"
			if self.file_stash.cache_get(bot_reply_key):
				logger.debug(f":: Submission already replied to by {bot}")
				bots_to_reply.pop()

		if len(bots_to_reply) == 0:
			return None

		text = None
		image = None
		if str(submission.url).endswith(('.png', '.jpg', '.jpeg')):
			logger.debug(f":: Submission contains image URL: {submission.url}")
			image = self.image_captioning.caption_image_from_url(submission.url)
		else:
			text = submission.selftext

		for bot in bots_to_reply:
			if str(submission.author).lower() == bot.lower():
				continue

			bot_reply_key = f"{bot}-{submission.id}"
			if self.file_stash.cache_get(bot_reply_key):
				logger.debug(f":: Submission already replied to by {bot}")
				continue

			personality = self.config.bot_map[bot]
			mapped_submission = {
				"subreddit": 'r/' + personality,
				"title": submission.title,
				"text": text,
				"image": image
			}
			if image is not None:
				constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|image|>{mapped_submission['image']}<|context_level|>0<|comment|>"
			else:
				constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}<|context_level|>0<|comment|>"

			self.file_stash.cache_set(bot_reply_key, True)
			data = {
				'text': constructed_string,
				'responding_bot': bot,
				'subreddit': mapped_submission['subreddit'],
				'reply_id': submission.id,
				'type': 'submission',
				'image': '',
				'title': mapped_submission['title']
			}
			logger.debug(f":: Sending Submission to queue for submission reply generation")
			self.file_queue.queue_put(data, QueueType.GENERATION)
