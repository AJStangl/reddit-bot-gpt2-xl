import os
import time
import warnings

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
from core.components.text.threaded_services.reply_sender import ReplyHandlerThread
from core.components.text.threaded_services.reply_generator import TextGenerationThread
from core.components.text.threaded_services.post_generation_thread import PostGenerationThread

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Bot(threading.Thread):
	def __init__(self, name: str, file_lock: threading.Lock, generative_services: GenerativeServices, file_queue: FileQueue):
		super().__init__(name=name, daemon=True)
		self.lock = file_lock
		self.file_stash: FileCache = FileCache(os.environ.get("CACHE_PATH"), self.lock)
		self.file_queue: FileQueue = file_queue
		self.generative_services: GenerativeServices = generative_services

		# Comments
		self.comment_handler_thread: CommentHandlerThread = CommentHandlerThread(name='comment-handler-thread', file_stash=self.file_stash, daemon=True, file_queue=self.file_queue)

		# Submissions
		self.submission_handler_thread: SubmissionHandlerThread = SubmissionHandlerThread(name='submission-handler-thread', file_stash=self.file_stash, daemon=True, file_queue=self.file_queue)

		# Shared Threads
		self.reply_handler_thread: ReplyHandlerThread = ReplyHandlerThread(name='reply-handler-thread', file_stash=self.file_stash, daemon=True, file_queue=self.file_queue)

		self.text_generator_thread: TextGenerationThread = TextGenerationThread(name='text-generation-thread', daemon=True, generative_services=self.generative_services, file_queue=self.file_queue)

		self.post_generation_thread: PostGenerationThread = PostGenerationThread(name='post-generation-thread', file_stash=self.file_stash, daemon=True, file_queue=self.file_queue)

		# Independent Thread
		self.queue_monitor_thread: QueueMonitorThread = QueueMonitorThread(name='queue-monitor-thread', daemon=True, file_queue=self.file_queue)


	def run(self):
		self.text_generator_thread.start()
		# Start process that monitors the queue
		self.queue_monitor_thread.start()
		# Start comment steam thread
		self.comment_handler_thread.start()
		# Start submission steam thread
		self.submission_handler_thread.start()
		# Start process that sends replies to reddit
		self.reply_handler_thread.start()
		# Start process that polls to create a post
		self.post_generation_thread.start()


if __name__ == '__main__':
	private_lock: threading.Lock = threading.Lock()
	private_file_queue: FileQueue = FileQueue()
	private_generative_services: GenerativeServices = GenerativeServices()
	bot_thread = Bot(name='bot-process', file_lock=private_lock, generative_services=private_generative_services, file_queue=private_file_queue)
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
