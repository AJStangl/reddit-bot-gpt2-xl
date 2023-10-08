import logging
import os
import threading
import time

import praw
import prawcore

from core.components.text.models.internal_types import QueueType
from core.components.text.services.file_queue_caching import FileCacheQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReplyHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue):
		threading.Thread.__init__(self, name=name)
		self.reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
		self.file_stash = file_stash
		self._stop_event = threading.Event()

	def run(self):
		logger.info(":: Starting Reply-Handler-Thread")
		self.process_reply_queue()

	def process_reply_queue(self):
		while True:
			try:
				self.handle_reply_queue()
				self.handle_post_queue()
			except Exception as e:
				logger.exception("Unexpected error: ", e)
			finally:
				time.sleep(5)

	def handle_reply_queue(self):
		data_thing: dict = self.file_stash.queue_pop(QueueType.REPLY)
		if data_thing is None:
			return
		if isinstance(data_thing, str):
			return
		if not isinstance(data_thing, dict):
			return
		self.process_reply(data_thing)

	def handle_post_queue(self):
		data_thing: dict = self.file_stash.queue_pop(QueueType.POST)
		if data_thing is None:
			return
		if isinstance(data_thing, dict):
			self.process_reply(data_thing)
		else:
			return

	def process_reply(self, data_thing):
		try:
			reply_text = data_thing.get("text")
			reply_bot = data_thing.get("responding_bot")
			reply_sub = data_thing.get("subreddit")
			reply_type = data_thing.get("type")
			reply_id = data_thing.get("reply_id")
			image = data_thing.get("image")

			if reply_text is None or reply_text == "":
				return

			new_reddit = praw.Reddit(site_name=reply_bot)

			if reply_type == 'submission':
				submission = new_reddit.submission(reply_id)
				reply = submission.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Submission: at https://www.reddit.com{reply.permalink}")

			if reply_type == 'comment':
				comment = new_reddit.comment(reply_id)
				reply = comment.reply(reply_text)
				logger.info(f":: {reply_bot} has replied to Comment: at https://www.reddit.com{reply.permalink}")

			if reply_type == 'post':
				if image is not None:
					new_reddit.subreddit(reply_sub).submit_image(title=reply_text, image_path=image)
				else:
					submission = new_reddit.subreddit(reply_sub).submit(title=reply_text, selftext=reply_text)
					logger.info(f":: {reply_bot} has posted to Subreddit: at https://www.reddit.com{submission.permalink}")

		except prawcore.PrawcoreException as e:
			logger.error(f"APIException: {e}")
			self.file_stash.queue_put(data_thing, QueueType.REPLY)

		except Exception as e:
			logger.error(f"An unexpected error occurred: {e}")
			time.sleep(5)