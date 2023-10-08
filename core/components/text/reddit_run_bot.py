import threading
from core.components.text.services.file_queue_caching import FileCacheQueue
from core.components.text.threaded_services.stream_comment import CommentHandlerThread
from core.components.text.threaded_services.stream_submission import SubmissionHandlerThread
from core.components.text.threaded_services.reply_sender import ReplyHandlerThread
from core.components.text.threaded_services.reply_generator import TextGenerationThread
from core.components.text.threaded_services.post_generation_thread import PostGenerationThread


class Bot(threading.Thread):
	def __init__(self, name: str, file_stash: FileCacheQueue):
		threading.Thread.__init__(self, name=name)
		self.file_stash: FileCacheQueue = file_stash

		# Polling
		self.comment_handler_thread: CommentHandlerThread = CommentHandlerThread(name='comment-handler-thread',
																				 file_stash=self.file_stash)

		self.submission_handler_thread: SubmissionHandlerThread = SubmissionHandlerThread(name='submission-handler-thread',
																						  file_stash=self.file_stash)

		# Replying
		self.reply_handler_thread: ReplyHandlerThread = ReplyHandlerThread(name='reply-handler-thread',
																		   file_stash=self.file_stash)

		# Submission Generation
		self.post_generation_thread: PostGenerationThread = PostGenerationThread(name='post-generation-thread',
																				 file_stash=self.file_stash)
		# Text Generation
		self.text_generation_thread: TextGenerationThread = TextGenerationThread(name='text-generation-thread',
																				 file_stash=self.file_stash)



		self._stop_event = threading.Event()

	def run(self):
		tasks = [
			self.post_generation_thread,
			self.reply_handler_thread,
			self.comment_handler_thread,
			self.submission_handler_thread,
			self.text_generation_thread,

		]
		[task.start() for task in tasks]
		while True:
			continue

