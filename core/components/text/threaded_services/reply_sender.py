import logging
import threading
import time
from typing import Optional

import praw

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


	def process_reply_queue(self) -> None:
		logger.info(":: Starting Reply-Handler-Thread")
		while True:
			try:
				self.handle_reply_queue()
			except Exception as e:
				logger.exception(e)
				time.sleep(5)
				continue

	def handle_reply_queue(self) -> None:
		data_thing: Optional[dict] = self.file_queue.queue_pop(QueueType.REPLY)
		if data_thing is None:
			return
		else:
			self.process_reply(data_thing)
			return

	def process_reply(self, data_thing) -> None:
		try:
			new_reddit = praw.Reddit(site_name=data_thing.get("responding_bot"))
			reply_text = data_thing.get("text")
			reply_bot = data_thing.get("responding_bot")
			reply_type = data_thing.get("type")
			reply_id = data_thing.get("reply_id")

			if reply_type == 'submission':
				submission = new_reddit.submission(reply_id)
				if submission.locked:
					return None
				reply = submission.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Submission: at https://www.reddit.com{reply.permalink}")
				return None

			if reply_type == 'comment':
				comment = new_reddit.comment(reply_id)
				if comment.submission.locked:
					return None
				if reply_text is None:
					return None
				if reply_text == "":
					reply_text = "I suppose I have nothing nice to say."
				reply = comment.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")
				return None

		except Exception as e:
			logger.exception(e)
			return None


class PostHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool, file_queue: FileQueue):
		super().__init__(name=name, daemon=daemon)
		self.file_stash: FileCache = file_stash
		self.file_queue: FileQueue = file_queue

	def run(self):
		logger.info(":: Starting Post-Handler-Thread")
		while True:
			try:
				self.handle_post_queue()
			except Exception as e:
				logger.exception(e)
				time.sleep(1)
				continue
			finally:
				time.sleep(60)

	def handle_post_queue(self):
		data_thing: dict = self.file_queue.queue_pop(QueueType.POST)
		if data_thing is None:
			time.sleep(1)
			return None
		if isinstance(data_thing, dict):
			self.process_reply(data_thing)
			time.sleep(1)
			return None
		else:
			time.sleep(1)
			return None


	def process_reply(self, data_thing) -> None:
		new_reddit = praw.Reddit(site_name=data_thing.get("responding_bot"))
		reply_text = data_thing.get("text")
		reply_title = data_thing.get("title")
		reply_bot = data_thing.get("responding_bot")
		reply_sub = data_thing.get("subreddit")
		image = data_thing.get("image")
		self.process_submission_post(reddit_instance=new_reddit, reply_sub=reply_sub, reply_bot=reply_bot, title=reply_title, text=reply_text, image=image)

	def process_submission_post(self, reddit_instance: praw.Reddit, reply_sub: str, reply_bot: str, title: str, text: str, image: str) -> None:
		if image is not None:
			submission = reddit_instance.subreddit(reply_sub).submit_image(title=title, image_path=image)
			logger.info(f":: {reply_bot} has posted an image: https://www.reddit.com{submission.permalink}")
			return None
		else:
			submission = reddit_instance.subreddit(reply_sub).submit(title=title, selftext=text)
			logger.info(f":: {reply_bot} has posted: https://www.reddit.com{submission.permalink}")
			return None