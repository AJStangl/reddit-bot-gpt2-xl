import multiprocessing
import os
import warnings
from multiprocessing import Process

from dotenv import load_dotenv

from core.components.text.threaded_services.queue_monitor_thread import QueueMonitorThread

warnings.filterwarnings("ignore")

load_dotenv()

import threading
import logging
import time

from core.components.text.services.file_queue_caching import FileCache
from core.components.text.threaded_services.stream_comment import CommentHandlerThread
from core.components.text.threaded_services.stream_submission import SubmissionHandlerThread
from core.components.text.threaded_services.reply_sender import ReplyHandlerThread
from core.components.text.threaded_services.reply_generator import TextGenerationThread
from core.components.text.threaded_services.post_generation_thread import PostGenerationThread

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextRunner:
	def __init__(self):
		self.text_generator = TextGenerationThread("text-generation-process", daemon=True)

	def run(self):
		self.text_generator.start()
		while True:
			try:
				time.sleep(1)
			except KeyboardInterrupt:
				self.text_generator.join()
				break


class Bot(threading.Thread):
	def __init__(self, name: str, lock: multiprocessing.Lock):
		super().__init__(name=name, daemon=True)
		# Singletons
		self.file_stash: FileCache = FileCache(os.environ.get("CACHE_PATH"), lock)
		# Threads
		self.comment_handler_thread: CommentHandlerThread = CommentHandlerThread(name='comment-handler-thread',
																				 file_stash=self.file_stash,
																				 daemon=True)
		self.submission_handler_thread: SubmissionHandlerThread = SubmissionHandlerThread(
			name='submission-handler-thread', file_stash=self.file_stash, daemon=True)

		# Shared Threads
		self.reply_handler_thread: ReplyHandlerThread = ReplyHandlerThread(name='reply-handler-thread',
																		   file_stash=self.file_stash, daemon=True)
		self.post_generation_thread: PostGenerationThread = PostGenerationThread(name='post-generation-thread',
																				 file_stash=self.file_stash,
																				 daemon=True)

		# Independent Thread
		self.queue_monitor_thread: QueueMonitorThread = QueueMonitorThread(name='queue-monitor-thread', daemon=True)

	def run(self):
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
	# Create main bot process
	lock: multiprocessing.Lock = multiprocessing.Lock()
	main_bot_thread = Bot(name='bot-process', lock=lock)
	bot_process = Process(target=main_bot_thread.run)

	# Create text generator process
	text_runner = TextRunner()
	text_gen_process = Process(target=text_runner.run)

	procs = [bot_process, text_gen_process]
	bot_process.run()
	text_gen_process.run()

	try:
		while True:
			continue
	except KeyboardInterrupt:
		for proc in procs:
			proc.terminate()
