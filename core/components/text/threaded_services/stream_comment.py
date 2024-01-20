import logging
import os
import random
import threading
import time
import json
import praw
import prawcore
from praw.models import Comment
from datetime import datetime
import pytz
import math
from core.components.text.models.internal_types import QueueType
from core.components.text.services.configuration_manager import ConfigurationManager
from core.components.text.services.file_queue_caching import FileCache, FileQueue

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommentHandlerThread(threading.Thread):
	def __init__(self, name: str, file_stash: FileCache, daemon: bool, file_queue: FileQueue, reddit: praw.Reddit):
		super().__init__(name=name, daemon=daemon)
		self.sub_names = os.environ.get("SUBREDDIT_TO_MONITOR")
		self.file_stash: FileCache = file_stash
		self.config = ConfigurationManager()
		self.file_queue = file_queue
		self.reddit: praw.Reddit = reddit

	def run(self):
		logger.info(":: Starting Comment-Handler-Thread")
		self.process_subreddit_stream()

	def decay_probability(self, created_utc):
		# Convert Reddit's created_utc to datetime object
		reddit_time = datetime.fromtimestamp(created_utc, pytz.utc)

		# Get current time in UTC
		current_time = datetime.now(pytz.utc)

		# Calculate time difference in hours
		difference = current_time - reddit_time
		difference_in_hours = difference.total_seconds() / 3600

		# Define lambda constant (1/4 for 4 hours)
		lambda_constant = 1 / 4

		# Calculate decay probability
		probability = math.exp(-lambda_constant * difference_in_hours)

		return probability

	def process_subreddit_stream(self):
		subreddit = self.reddit.subreddit(self.sub_names)
		while True:
			try:
				self.process_comments_in_stream(subreddit)
			except prawcore.exceptions.TooManyRequests as e:
				logger.exception(e)
				time.sleep(10)
				continue
			except Exception as e:
				logger.exception(e)
				time.sleep(10)
				continue

	def process_comments_in_stream(self, subreddit):
		for item in subreddit.stream.comments(pause_after=0, skip_existing=True):
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
			return
		else:
			self.process_comment(comment=comment)
			self.file_stash.cache_set(item_key, True)
			return

	def process_comment(self, comment: Comment):
		if comment is None:
			return

		submission_id = comment.submission
		bots = list(self.config.bot_map.keys())
		filtered_bot = [x for x in bots if x.lower() != str(comment.author).lower()]
		random.shuffle(filtered_bot)
		for i, bot in enumerate(filtered_bot):
			if random.randint(0, 1) == 0:
				continue

			if random.randint(0, 1) == 1:
				continue

			submission: praw.models.Submission = self.reddit.submission(submission_id)

			chance_of_reply_if_filtered = 0.005  # .5% chance
			chance_of_reply_if_not_filtered = 0.05  # 5% chance

			# Convert the subreddit to lowercase and compare
			if str(submission.subreddit).lower() == 'subsimGPT2interactive'.lower():
				# Check if the bot is in the filtered list
				if bot in filtered_bot: # or 'gpt' in str(bot).lower():
					# Use a random float to determine if the bot should reply
					if random.random() < chance_of_reply_if_filtered:
						# Code to reply to the submission
						pass
				else:
					# If the bot is not filtered, use the other chance variable
					if random.random() < chance_of_reply_if_not_filtered:
						# Code to reply to the submission
						pass

			reply_probability = self.decay_probability(comment.created_utc)

			if reply_probability < 0.1:
				continue

			personality = self.config.bot_map[bot]
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
				'responding_bot': bot,
				'subreddit': mapped_submission['subreddit'],
				'reply_id': comment.id,
				'type': 'comment'
			}

			if str(submission.subreddit).lower() == 'subsimGPT2interactive'.lower():
				with open(f"/mnt/m/home/workspaces/machine_learning/messages/{comment.id}.json", 'w') as f:
					data_string = json.dumps(data)
					f.write(data_string)
			else:
				self.file_queue.queue_put(data, QueueType.GENERATION)
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
					try:
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
						time.sleep(30)
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
