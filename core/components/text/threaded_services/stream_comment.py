import logging
import os
import random
import threading
import time

import praw
import prawcore
from praw.models import Comment

from core.components.text.models.internal_types import QueueType
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCacheQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommentHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue, daemon: bool):
		threading.Thread.__init__(self, name=name, daemon=daemon)
		self.reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
		self.sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		self.file_stash: FileCacheQueue = file_stash
		self.config = ConfigurationManager()

	def run(self):
		logger.info(":: Starting Comment-Handler-Thread")
		subreddit = self.reddit.subreddit(self.sub_names)
		self.process_subreddit_stream(subreddit)

	def process_subreddit_stream(self, subreddit):
		while True:
			try:
				self.process_comments_in_stream(subreddit)
			except Exception as e:
				logger.exception(":: Unexpected error in process_subreddit_stream", e)
				time.sleep(5)
				continue

	def process_comments_in_stream(self, subreddit):
		for item in subreddit.stream.comments(pause_after=0, skip_existing=True):
			if random.random() > 0.5:
				continue
			if item is None:
				time.sleep(1)
				continue

			if isinstance(item, Comment):
				self.handle_comment(item)

	def handle_comment(self, comment: Comment):
		item_key = f"{comment.id}-comment"
		item_seen = self.file_stash.cache_get(item_key)

		if item_seen:
			time.sleep(5)
		else:
			self.process_comment(comment=comment)
			self.file_stash.cache_set(item_key, True)

	def process_comment(self, comment: Comment):
		if comment is None:
			return

		submission_id = comment.submission
		bots = list(self.config.bot_map.keys())
		filtered_bot = [x for x in bots if x.lower() != str(comment.author).lower()]
		responding_bot = random.choice(filtered_bot)
		personality = random.choice(self.config.personality_list)
		submission = self.reddit.submission(submission_id)
		mapped_submission = {
			"subreddit": 'r' + '/' + personality,
			"title": submission.title,
			"text": submission.selftext
		}

		if int(submission.num_comments) > int(os.environ.get('MAX_REPLIES')):
			logger.debug(f":: Comment Has More Than 250 Replies, Skipping")
			self.file_stash.cache_set(comment.id, True)
			return

		constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
		constructed_string += self.construct_context_string(comment)
		data = {
			'text': constructed_string,
			'responding_bot': responding_bot,
			'subreddit': mapped_submission['subreddit'],
			'reply_id': comment.id,
			'type': 'comment'
		}
		self.file_stash.queue_put(data, QueueType.GENERATION)
		self.file_stash.cache_set(comment.id, True)

	def construct_context_string(self, comment: Comment) -> str:
		things = []
		current_comment = comment
		counter = 0
		try:
			while not isinstance(current_comment, praw.models.Submission):
				comment_key = str(current_comment.id) + "-" + 'text'
				cached_thing = self.file_stash.cache_get(comment_key)
				if cached_thing is not None:
					cached = {
						'counter': counter,
						'text': cached_thing
					}
					if isinstance(cached, dict):
						cached = cached.get('text')
						things.append(cached)
					else:
						things.append(cached)
					counter += 1
					current_comment = current_comment.parent()
					continue
				else:
					thing = {
						"text": "",
						"counter": 0
					}
					if thing is None:
						time.sleep(1)
						current_comment = current_comment.parent()
						continue
					thing['counter'] = counter
					thing['text'] = current_comment.body
					things.append(current_comment.body)
					self.file_stash.cache_set(comment_key, current_comment.body)
					counter += 1
					if counter == 8:
						break
					else:
						time.sleep(1)
						current_comment = current_comment.parent()
						continue
		except prawcore.exceptions.RequestException as request_exception:
			logger.exception("Request Error", request_exception)
			time.sleep(5)
		except Exception as e:
			logger.exception(f"General Exception In construct_context_string", e)
			time.sleep(5)

		things.reverse()
		out = ""
		for i, r in enumerate(things):
			out += f"<|context_level|>{i}<|comment|>{r}"

		out += f"<|context_level|>{len(things)}<|comment|>"
		return out
