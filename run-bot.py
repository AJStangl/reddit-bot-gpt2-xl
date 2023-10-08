import logging
import signal
import time
import warnings
import os

from dotenv import load_dotenv

from core.components.text.services.file_queue_caching import FileCacheQueue

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
		self.comment_handler_thread:    CommentHandlerThread = CommentHandlerThread(name='comment-handler-thread', file_stash=self.file_stash)
		self.submission_handler_thread: SubmissionHandlerThread = SubmissionHandlerThread(name='submission-handler-thread', file_stash=self.file_stash)
		self.reply_handler_thread:      ReplyHandlerThread = ReplyHandlerThread(name='reply-handler-thread', file_stash=self.file_stash)
		self.post_generation_thread:    PostGenerationThread = PostGenerationThread(name='post-generation-thread', file_stash=self.file_stash)
		self.text_generation_thread:    TextGenerationThread = TextGenerationThread(name='text-generation-thread', file_stash=self.file_stash)

	def run(self):
		self.comment_handler_thread.start()
		self.submission_handler_thread.start()
		self.reply_handler_thread.start()
		self.post_generation_thread.start()
		self.text_generation_thread.start()


if __name__ == '__main__':
	file_stash: FileCacheQueue = FileCacheQueue(os.environ.get("CACHE_PATH"))
	main_bot_thread: Bot = Bot(name='reddit-bot', file_stash=file_stash)
	main_bot_thread.start()

	while True:
		try:
			time.sleep(1)
			continue
		except KeyboardInterrupt:
			main_bot_thread.join()
