import json
import shelve
from enum import Enum
from pathlib import Path
from typing import Optional
import os
import random
import praw
import prawcore
import requests
import torch
from PIL import Image
from dotenv import load_dotenv
from praw.models import Submission, Comment
from torch import Tensor
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BertTokenizer
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import logging
import time

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubmissionDetails(object):
	def __init__(self):
		self.id: str = None
		self.author: str = None
		self.subreddit: str = None
		self.title: str = None
		self.text: str = None
		self.score: int = None
		self.comments: list = None
		self.type: SubmissionType = None


class CommentDetails(object):
	def __init__(self):
		self.id: str = None
		self.author: str = None
		self.text: str = None
		self.replies: dict = None
		self.parent_id: dict = None
		self.link_id: dict = None
		self.context_level: int = None


class SubmissionType(Enum):
	image = "<|image|>"
	gallery = "<|gallery|>"
	video = "<|video|>"
	comment = "<|comment|>"
	text = "<|text|>"
	crosspost = "<|crosspost|>"
	link = "<|link|>"
	unknown = None


class Blip(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained("D:\\models\\blip-captioning\\blip").to(self.device)
		self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

	def load_image(self, image_url, image_size=384, device="cuda") -> Optional[Tensor]:
		try:
			raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
			w, h = raw_image.size
			transform = transforms.Compose([
				transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
			])
			image = transform(raw_image).unsqueeze(0).to(device)
			return image
		except Exception as e:
			logger.exception(e)
			return None

	def caption_image(self, image_url) -> Optional[str]:
		try:
			image_size = 384
			image = self.load_image(image_url=image_url, image_size=image_size, device=self.device)
			with torch.no_grad():
				input_ids = self.tokenizer(["a picture of"], return_tensors="pt").input_ids.to(self.device)
				output_ids = self.model.generate(image, input_ids)
				caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
				caption_new = caption.replace('[UNK]', '').strip()
				return caption_new
		except Exception as e:
			logger.exception(e)
			return None


class Utility:
	@staticmethod
	def identify_submission_type(submission: Submission) -> SubmissionType:
		is_image = any([submission.url.endswith(item) for item in ['.jpg', '.jpeg', '.png']])
		if is_image:
			return SubmissionType.image
		if hasattr(submission, 'is_gallery') and submission.is_gallery:
			return SubmissionType.gallery
		if hasattr(submission, 'is_video') and submission.is_video:
			return SubmissionType.video
		if hasattr(submission, 'is_self') and submission.is_self:
			return SubmissionType.text
		if hasattr(submission, 'selftext') and submission.selftext:
			return SubmissionType.text
		if hasattr(submission, 'crosspost_parent_list'):
			return SubmissionType.crosspost
		else:
			return SubmissionType.unknown

	@staticmethod
	def get_blacklist_authors() -> dict:
		authors: dict = {'AmputatorBot', 'analyzeHistory', 'anti-gif-bot', 'AnimalFactsBot', 'automoderator',
						 'autotldr',
						 'auto-xkcd37', 'autourbanbot', 'AyyLmao2DongerBot-v2', 'backtickbot', 'BadDadBot',
						 'BaseballBot',
						 'b0trank', 'Bot_Metric', 'CakeDay--Bot', 'checks_out_bot', 'ClickableLinkBot',
						 'CodeFormatHelperBot', 'CoolDownBot', 'CommonMisspellingBot', 'converter-bot',
						 'could-of-bot',
						 'DailMail_Bot', '[deleted]', 'EmojifierBot', 'enzo32ferrari', 'exponant',
						 'fast-parenthesis-bot',
						 'FatFingerHelperBot', 'FlairHelperBot', 'Freedom_Unit_Bot', 'friendly-bot', 'fukramosbot',
						 'GenderNeutralBot', 'gfy_mirror', 'gifv-bot', 'GitCommandBot', 'GitHubPermalinkBot',
						 'Gyazo_Bot',
						 'GoodBot_BadBot', 'haikubot-1911', 'haikusbot', 'HelperBot_', 'highlightsbot',
						 'HuachiBot',
						 'IamYodaBot', 'i-am-dad-bot', 'imguralbumbot', 'ImJimmieJohnsonBot', 'Its_URUGUAY_bot',
						 'JobsHelperBot', 'JustAHooker', 'kmcc93', 'LinkFixerBot', 'LinkifyBot', 'link-reply-bot',
						 'LearnProgramming_Bot', 'LimbRetrieval-Bot', 'LinkExpanderBot', 'MAGIC_EYE_BOT',
						 'MaxImageBot',
						 'Mentioned_Videos', 'metric_units', 'MLBVideoConverterBot', 'ModeratelyHelpfulBot',
						 'morejpeg_auto', 'NASCARThreadBot', 'NBA_MOD', 'NFL_Warning', 'NFLVideoConverterBot',
						 'nice-scores', 'NicolasBotCage', 'Not_RepostSleuthBot', 'of_have_bot', 'ootpbot',
						 'originalpostsearcher', 'oofed-bot', 'parenthesis-bot', 'PicDescriptionBot',
						 'phonebatterylevelbot', 'PORTMANTEAU-BOT', 'ProgrammerHumorMods', 'BeginnerProjectBot',
						 'pythonHelperBot', 'reddit-blackjack-bot', 'Reddit-Book-Bot', 'redditstreamable',
						 'relevant_post_bot', 'remindmebot', 'repliesnice', 'RepostSleuthBot', 'RepostCheckerBot',
						 'ReverseCaptioningBot', 'roastbot', 'RoastBotTenThousand', 'sexy-snickers',
						 'should_have_listened', 'Simultate_Me_Bot', 'SmallSubBot', 'SnapshillBot', 'sneakpeekbot',
						 'Spam_Detector_Bot', 'Shakespeare-Bot', 'SpellCheck_Privilege', 'StreamableReddit',
						 'streamablemirrors', 'sub_doesnt_exist_bot', 'SwagmasterEDP', 'table_it_bot',
						 'thank_mr_skeltal_bot', 'Thatonefriendlybot', 'THE_GREAT_SHAZBOT', 'TheDroidNextDoor',
						 'timezone_bot', 'Title2ImageBot', 'TitleToImageBot', 'totesmessenger',
						 'twittertostreamable',
						 'tweetposter', 'TweetsInCommentsBot', 'tweettranscriberbot', 'twitterInfo_bot',
						 'TwitterVideoBot',
						 'User_Simulator', 'vredditdownloader', 'video_descriptionbot', 'WaterIsWetBot',
						 'WellWishesBot',
						 'WikiTextBot', 'WikiSummarizerBot', 'xkcd-Hyphen-bot', 'xkcd_transcriber',
						 'YoMammaJokebot',
						 'youtubefactsbot', 'YTubeInfoBot', 'sneakpeekbot','G2Minion'}
		return {author.lower() for author in authors}

	@staticmethod
	def is_blacklisted(author: str) -> bool:
		return author.lower() in Utility.get_blacklist_authors()

	@staticmethod
	def is_remove_or_deleted(input_string: str):
		if input_string == "[removed]" or input_string == "[deleted]":
			return True
		else:
			return False


class DataBuilder:
	@staticmethod
	def collect_submission_details(submission: Submission, image_processor: Blip):
		try:
			author = str(submission.author).lower()
			submission_type: SubmissionType = Utility.identify_submission_type(submission=submission)
			if submission_type == SubmissionType.unknown:
				return None

			if Utility.is_blacklisted(author):
				return None

			details = SubmissionDetails()
			details.id = submission.id
			details.author = str(submission.author)
			details.subreddit = "r/" + str(submission.subreddit)
			details.title = str(submission.title)
			details.score = int(submission.score)
			details.comments = []
			details.type = submission_type

			if submission_type == SubmissionType.text:
				details.text = str(submission.selftext)
			if submission_type == SubmissionType.image:
				caption = image_processor.caption_image(submission.url)
				details.text = caption
			if submission_type == SubmissionType.video:
				thumbnail = submission.thumbnail
				caption = image_processor.caption_image(thumbnail)
				details.text = caption
			if submission_type == SubmissionType.gallery:
				captions = DataBuilder.handle_gallery(submission, image_processor)
				all_captions = [item for item in captions]
				joined_captions = ",".join(all_captions).strip()
				details.text = joined_captions
			if submission_type == SubmissionType.crosspost:
				parent_permalink = submission.crosspost_parent_list[0].get('permalink', '')
				parent_url = f"https://www.reddit.com{parent_permalink}"
				details.text = parent_url
			if submission_type == SubmissionType.link:
				details.text = str(submission.url)
			if submission_type == SubmissionType.comment:
				pass

			return details

		except Exception as e:
			logger.error(e)
			return None

	@staticmethod
	def handle_gallery(submission: Submission, image_processor: Blip):
		images = DataBuilder.get_gallery_images(submission)
		captions = DataBuilder.caption_images(images, image_processor)
		return captions

	@staticmethod
	def get_gallery_images(submission: Submission) -> list:
		gallery_images = []
		if hasattr(submission, 'is_gallery') and submission.is_gallery:
			media_metadata = submission.media_metadata
			if media_metadata is None:
				return gallery_images

			if media_metadata.items() is None:
				return gallery_images

			for item_id, item_data in media_metadata.items():
				image_url = item_data['s']['u']
				gallery_images.append(image_url)

		return gallery_images

	@staticmethod
	def caption_images(image_urls: list, image_processor: Blip) -> list:
		captions = []
		try:
			for image_url in image_urls:
				caption = image_processor.caption_image(image_url)
				if caption is not None:
					captions.append(caption)
		except Exception as e:
			logger.error(e)
		return captions

	@staticmethod
	def get_all_comments(submission: Submission):
		try:
			submission.comments.replace_more(limit=None)
		except prawcore.exceptions.TooManyRequests as e:
			logger.error(e)
			time.sleep(60)
			logger.info("Retrying...")
			submission.comments.replace_more(limit=None)
		except Exception as e:
			logger.error(e)
			raise e

		root_comments = submission.comments  # Removed await

		comment_tree = {}
		for comment in tqdm(root_comments, total=len(root_comments), desc="Processing comments"):
			author = str(comment.author).lower()
			body = comment.body
			if Utility.is_blacklisted(author):
				continue
			if Utility.is_remove_or_deleted(body):
				continue
			comment_details: dict = DataBuilder.collect_comments(comment)
			comment_tree[comment.id] = comment_details
		return comment_tree

	@staticmethod
	def collect_comments(comment: Comment, context_level=0) -> CommentDetails:
		comment_details = CommentDetails()
		comment_details.id = str(comment.id)
		comment_details.author = str(comment.author)
		comment_details.text = comment.body
		comment_details.replies = {}
		comment_details.parent_id = str(comment.parent_id)
		comment_details.link_id = str(comment.link_id)
		comment_details.context_level = context_level
		comment_replies = comment.replies  # Removed await

		for reply in comment_replies:
			reply_details = DataBuilder.collect_comments(reply, context_level + 1)
			comment_details.replies[reply.id] = reply_details

		return comment_details

	@staticmethod
	def extract_comments(comment_tree, current_str, strings):
		for comment_id, comment_details in comment_tree.items():
			context_level = comment_details.context_level
			body = comment_details.text
			new_str = current_str + f'<|context_level|>{context_level}<|comment|>{body}'
			if comment_details.replies:
				DataBuilder.extract_comments(comment_details.replies, new_str, strings)
			else:
				strings.append(new_str + '<|endoftext|>')


def main():
	image_processor: Blip = Blip()
	script_dir = Path(__file__).resolve().parent
	cache_path = os.path.join(script_dir, 'cache')
	os.makedirs(cache_path, exist_ok=True)
	cache_db_path = Path(cache_path, "cache.db")
	site_name = os.environ.get("REDDIT_ACCOUNT_SECTION_NAME_2")
	reddit = praw.Reddit(site_name=site_name)
	# while True:
	# 	with shelve.open(str(cache_db_path)) as db:
	# 		try:
	# 			subs = os.environ.get("SUBS_TO_MINE").split(",")
	# 			random.shuffle(subs)
	# 			for sub in subs:
	# 				subreddit = reddit.subreddit(sub)
	# 				submissions = list(subreddit.top(time_filter="all", limit=1000))
	# 				for submission in tqdm(submissions, total=len(submissions), desc=f"Processing submissions for: {sub}"):
	while True:
		with shelve.open(str(cache_db_path)) as db:
			try:
				subreddit = reddit.subreddit('all')  # 'all' subreddit gathers posts from all public subreddits.
				for comment in subreddit.stream.comments(skip_existing=True):  # Iterates over comments as they become available.
					submission = comment.submission
					try:
						sub_type: SubmissionType = Utility.identify_submission_type(submission)
						if sub_type == SubmissionType.text:
							pass
						if sub_type == SubmissionType.image:
							pass
						else:
							continue

						if submission.id in db:
							continue

						basic_submission: SubmissionDetails = DataBuilder.collect_submission_details(submission, image_processor)
						if basic_submission is None:
							continue
						try:
							comment_tree = DataBuilder.get_all_comments(submission)
							basic_submission.comments = comment_tree
						except prawcore.exceptions.TooManyRequests as e:
							logger.error(e)
							time.sleep(10)
							comment_tree = DataBuilder.get_all_comments(submission)
							basic_submission.comments = comment_tree
							continue
						except Exception as e:
							logger.error(e)
							continue

						constructed_strings = []
						submission_type: str = str(basic_submission.type.value)

						base_str = f'<|startoftext|><|subreddit|>{basic_submission.subreddit}<|title|>{basic_submission.title}{submission_type}{basic_submission.text}'
						DataBuilder.extract_comments(basic_submission.comments, base_str, constructed_strings)

						data_path = os.path.join(script_dir, "data")
						os.makedirs(data_path, exist_ok=True)
						filename = Path(data_path, f"{submission.id}.txt")
						filename.touch(exist_ok=True)

						with filename.open("wb") as f:
							for item in constructed_strings:
								encoded = item.encode('unicode_escape')
								f.write(encoded)
								f.write(b'\n')
						db[submission.id] = True
						logging.info(f"Successfully processed and saved submission {submission.id}")
					finally:
						time.sleep(1)
			except Exception as e:
				logger.error(f"An error occurred: {e}", exc_info=True)
				continue