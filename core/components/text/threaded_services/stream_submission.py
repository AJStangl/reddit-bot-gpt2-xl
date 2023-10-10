import logging
import os
import random
import threading
import time

import praw
from praw.models import Submission

from core.components.text.models.internal_types import QueueType
from core.components.text.models.queue_message import RedditComment
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCache, FileQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubmissionHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool):
		super().__init__(name=name, daemon=daemon)
		self.reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
		self.sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		self.file_stash: FileCache = file_stash
		self.file_queue: FileQueue = FileQueue()
		self.config = ConfigurationManager()

	def run(self):
		logger.info(":: Starting Submission-Handler-Thread")
		subreddit = self.reddit.subreddit(self.sub_names)
		self.process_subreddit_stream(subreddit)

	def process_subreddit_stream(self, subreddit: praw):
		while True:
			try:
				self.process_submissions_in_stream(subreddit)
			except Exception as e:
				logger.exception(e)
				time.sleep(1)

	def process_submissions_in_stream(self, subreddit):
		for item in subreddit.stream.submissions(pause_after=0, skip_existing=False):
			if item is None:
				time.sleep(60)
				continue

			if isinstance(item, Submission):
				self.handle_submission(item)

	def handle_submission(self, submission: Submission):
		item_key = f"{submission.id}-submission"
		item_seen = self.file_stash.cache_get(item_key)

		if item_seen:
			time.sleep(5)
		else:
			self.process_submission(submission=submission)
			self.file_stash.cache_set(item_key, True)

	def process_submission(self, submission):
		if submission is None:
			time.sleep(1)
			return

		bots_to_reply = list(self.config.bot_map.keys())
		for bot in self.config.bot_map.keys():
			bot_reply_key = f"{bot}-{submission.id}"
			if self.file_stash.cache_get(bot_reply_key):
				bots_to_reply.pop()

		if len(bots_to_reply) == 0:
			return None

		if str(submission.url).endswith(('.png', '.jpg', '.jpeg')):
			logger.debug(f":: Submission contains image URL: {submission.url}")
			text = None

		else:
			text = submission.selftext

		for bot in bots_to_reply:
			if str(submission.author).lower() == bot.lower():
				continue
			personality = random.choice(self.config.personality_list)
			mapped_submission = {
				"subreddit": 'r/' + personality,
				"title": submission.title,
				"text": text
			}
			constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}<|context_level|>0<|comment|>"
			bot_reply_key = f"{bot}-{submission.id}"
			bot_reply_value = self.file_stash.cache_get(bot_reply_key)
			if bot_reply_value:
				continue
			else:
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
				reddit_data = RedditComment(**data)
				reddit_json = reddit_data.to_dict()
				self.file_queue.queue_put(data, QueueType.GENERATION)
