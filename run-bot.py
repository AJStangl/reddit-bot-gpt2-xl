import os
import time
import warnings

import praw
from dotenv import load_dotenv

from core.components.text.services.text_generation import GenerativeServices
from core.components.text.threaded_services.queue_monitor_thread import QueueMonitorThread

warnings.filterwarnings("ignore")

load_dotenv()

import threading
import logging

from core.components.text.services.file_queue_caching import FileCache, FileQueue
from core.components.text.threaded_services.stream_comment import CommentHandlerThread
from core.components.text.threaded_services.stream_submission import SubmissionHandlerThread
from core.components.text.threaded_services.reply_sender import ReplyHandlerThread, PostHandlerThread
from core.components.text.threaded_services.reply_generator import TextGenerationThread
from core.components.text.threaded_services.post_generation_thread import PostGenerationThread

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Bot(threading.Thread):
	def __init__(self, name: str, file_lock: threading.Lock, generative_services: GenerativeServices, file_queue: FileQueue, reddit: praw.Reddit):
		super().__init__(name=name, daemon=True)
		self.lock = file_lock
		self.reddit = reddit

		self.file_stash: FileCache = FileCache(os.environ.get("CACHE_PATH"), self.lock)

		self.file_queue: FileQueue = file_queue

		self.generative_services: GenerativeServices = generative_services

		self.comment_handler_thread:    CommentHandlerThread = (
			CommentHandlerThread(name='comment-handler-thread', daemon=True, file_stash=self.file_stash, file_queue=self.file_queue, reddit=self.reddit))

		self.submission_handler_thread: SubmissionHandlerThread = (
			SubmissionHandlerThread(name='submission-handler-thread', daemon=True, file_stash=self.file_stash,  file_queue=self.file_queue, reddit=self.reddit))

		self.reply_handler_thread:      ReplyHandlerThread = (
			ReplyHandlerThread(name='reply-handler-thread', daemon=True, file_stash=self.file_stash, file_queue=self.file_queue))

		self.text_generator_thread:     TextGenerationThread = (
			TextGenerationThread(name='text-generation-thread', daemon=True, file_queue=self.file_queue, generative_services=self.generative_services))

		self.post_generation_thread:    PostGenerationThread = (
			PostGenerationThread(name='post-generation-thread', daemon=True, file_stash=self.file_stash, file_queue=self.file_queue))

		self.post_handler_thread: PostHandlerThread = (
			PostHandlerThread(name='post-handler-thread', daemon=True, file_stash=self.file_stash, file_queue=self.file_queue)
		)

		self.queue_monitor_thread: QueueMonitorThread = (
			QueueMonitorThread(name='queue-monitor-thread', daemon=True, file_queue=self.file_queue))


	def run(self):
		self.text_generator_thread.start()
		self.queue_monitor_thread.start()
		self.comment_handler_thread.start()
		self.submission_handler_thread.start()
		self.reply_handler_thread.start()
		self.post_handler_thread.start()
		self.post_generation_thread.start()


if __name__ == '__main__':
	lock_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\locks"
	for lock in os.listdir(lock_path):
		os.remove(os.path.join(lock_path, lock))
	private_lock: threading.Lock = threading.Lock()
	private_file_queue: FileQueue = FileQueue()
	private_generative_services: GenerativeServices = GenerativeServices()
	private_reddit: praw.Reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))
	bot_thread: Bot = Bot(name='bot-process', file_lock=private_lock, generative_services=private_generative_services, file_queue=private_file_queue, reddit=private_reddit)
	bot_thread.run()

	try:
		while True:
			time.sleep(1)
			if not bot_thread.text_generator_thread.is_alive():
				exit(1)
			continue
	except KeyboardInterrupt:
		logger.info('Shutdown')
		exit(0)
