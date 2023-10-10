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

from core.components.text.services.file_queue_caching import FileCacheQueue
from core.components.text.threaded_services.stream_comment import CommentHandlerThread
from core.components.text.threaded_services.stream_submission import SubmissionHandlerThread
from core.components.text.threaded_services.reply_sender import ReplyHandlerThread
from core.components.text.threaded_services.reply_generator import TextGenerationThread
from core.components.text.threaded_services.post_generation_thread import PostGenerationThread

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Bot(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue):
		threading.Thread.__init__(self, name=name)
		self.file_stash: FileCacheQueue = file_stash
		self.comment_handler_thread: CommentHandlerThread = CommentHandlerThread(name='comment-handler-thread',file_stash=self.file_stash,daemon=True)
		self.submission_handler_thread: SubmissionHandlerThread = SubmissionHandlerThread(name='submission-handler-thread', file_stash=self.file_stash, daemon=True)
		self.reply_handler_thread: ReplyHandlerThread = ReplyHandlerThread(name='reply-handler-thread', file_stash=self.file_stash, daemon=True)
		self.post_generation_thread: PostGenerationThread = PostGenerationThread(name='post-generation-thread', file_stash=self.file_stash,daemon=True)
		self.text_generation_thread: TextGenerationThread = TextGenerationThread(name='text-generation-thread', file_stash=self.file_stash,daemon=True)
		self.queue_monitor_thread: QueueMonitorThread = QueueMonitorThread(name='queue-monitor-thread', file_stash=self.file_stash, daemon=True)

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
		# Start process that generates text
		self.text_generation_thread.start()


if __name__ == '__main__':
	internal_file_stash: FileCacheQueue = FileCacheQueue(os.environ.get("CACHE_PATH"))
	bot_process = Bot(name='bot-process', file_stash=internal_file_stash)
	bot_process.start()

	while True:
		try:
			time.sleep(1)
		except KeyboardInterrupt:
			break
