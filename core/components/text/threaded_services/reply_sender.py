import logging
import os
import threading
import time

import praw
import prawcore
from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileCache, FileQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReplyHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool, file_queue: FileQueue):
		super().__init__(name=name, daemon=daemon)
		self.file_stash: FileCache = file_stash
		self.file_queue: FileQueue = file_queue

	def run(self):
		self.process_reply_queue()

	def process_reply_queue(self):
		logger.info(":: Starting Reply-Handler-Thread")
		while True:
			try:
				self.handle_reply_queue()
				self.handle_post_queue()
			except Exception as e:
				logger.exception("Unexpected error: ", e)
				continue
			finally:
				time.sleep(5)

	def handle_reply_queue(self):
		data_thing: dict = self.file_queue.queue_pop(QueueType.REPLY)
		if data_thing is None:
			return
		if isinstance(data_thing, str):
			return
		if not isinstance(data_thing, dict):
			return
		self.process_reply(data_thing)

	def handle_post_queue(self):
		data_thing: dict = self.file_queue.queue_pop(QueueType.POST)
		if data_thing is None:
			return
		if isinstance(data_thing, dict):
			self.process_reply(data_thing)
		else:
			return

	def process_reply(self, data_thing):
		try:
			new_reddit = praw.Reddit(site_name=data_thing.get("responding_bot"))
			reply_text = data_thing.get("text")
			reply_bot = data_thing.get("responding_bot")
			reply_sub = data_thing.get("subreddit")
			reply_type = data_thing.get("type")
			reply_id = data_thing.get("reply_id")
			image = data_thing.get("image")

			if reply_type == 'submission':
				submission = new_reddit.submission(reply_id)
				reply = submission.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Submission: at https://www.reddit.com{reply.permalink}")

			if reply_type == 'comment':
				comment = new_reddit.comment(reply_id)
				reply = comment.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")

			if reply_type == 'post':
				self.process_submission_post(reddit_instance=new_reddit, reply_sub=reply_sub, reply_bot=reply_bot, title=reply_text, text=reply_text)

		except prawcore.PrawcoreException as e:
			logger.error(f"APIException: {e}")
			self.file_queue.queue_put(data_thing, QueueType.REPLY)

		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")
			time.sleep(5)

	def process_submission_post(self, reddit_instance: praw.Reddit, reply_sub: str, reply_bot: str, title: str, text: str):
			try:
				submission = reddit_instance.subreddit(reply_sub).submit(title=title, selftext=text)
				logger.info(f":: {reply_bot} has posted to Subreddit: at https://www.reddit.com{submission.permalink}")
			except Exception as e:
				logger.exception(e)