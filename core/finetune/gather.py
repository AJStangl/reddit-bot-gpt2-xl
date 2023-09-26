import json
import time
from pathlib import Path
import asyncpraw
import asyncprawcore
from asyncpraw.models import Comment
import os
import asyncio

from dotenv import load_dotenv
from tqdm import tqdm
import shelve

import logging
from io import BytesIO
from typing import AsyncGenerator

import aiohttp
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CaptionProcessor(object):
	def __init__(self, device_name: str = "cuda"):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.device_name = device_name
		self.device = torch.device(self.device_name)
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

	async def caption_image_from_url(self, image_url: str) -> AsyncGenerator:
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(image_url) as response:
					if response.status != 200:
						yield None
					content = await response.read()
					image = Image.open(BytesIO(content))
					try:
						inputs = self.processor(images=image, return_tensors="pt").to(self.device)
						out = self.model.generate(**inputs, max_new_tokens=77, num_return_sequences=1, do_sample=True)
						yield self.processor.decode(out[0], skip_special_tokens=True)
					except Exception as e:
						logger.exception(e)
						yield None
					finally:
						image.close()
		except Exception as e:
			logger.exception(e)
			yield None



# Create the set of blacklisted authors once
BLACK_LIST_AUTHORS = {'AmputatorBot', 'analyzeHistory', 'anti-gif-bot', 'AnimalFactsBot', 'automoderator', 'autotldr',
					  'auto-xkcd37', 'autourbanbot', 'AyyLmao2DongerBot-v2', 'backtickbot', 'BadDadBot', 'BaseballBot',
					  'b0trank', 'Bot_Metric', 'CakeDay--Bot', 'checks_out_bot', 'ClickableLinkBot',
					  'CodeFormatHelperBot', 'CoolDownBot', 'CommonMisspellingBot', 'converter-bot', 'could-of-bot',
					  'DailMail_Bot', '[deleted]', 'EmojifierBot', 'enzo32ferrari', 'exponant', 'fast-parenthesis-bot',
					  'FatFingerHelperBot', 'FlairHelperBot', 'Freedom_Unit_Bot', 'friendly-bot', 'fukramosbot',
					  'GenderNeutralBot', 'gfy_mirror', 'gifv-bot', 'GitCommandBot', 'GitHubPermalinkBot', 'Gyazo_Bot',
					  'GoodBot_BadBot', 'haikubot-1911', 'haikusbot', 'HelperBot_', 'highlightsbot', 'HuachiBot',
					  'IamYodaBot', 'i-am-dad-bot', 'imguralbumbot', 'ImJimmieJohnsonBot', 'Its_URUGUAY_bot',
					  'JobsHelperBot', 'JustAHooker', 'kmcc93', 'LinkFixerBot', 'LinkifyBot', 'link-reply-bot',
					  'LearnProgramming_Bot', 'LimbRetrieval-Bot', 'LinkExpanderBot', 'MAGIC_EYE_BOT', 'MaxImageBot',
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
					  'timezone_bot', 'Title2ImageBot', 'TitleToImageBot', 'totesmessenger', 'twittertostreamable',
					  'tweetposter', 'TweetsInCommentsBot', 'tweettranscriberbot', 'twitterInfo_bot', 'TwitterVideoBot',
					  'User_Simulator', 'vredditdownloader', 'video_descriptionbot', 'WaterIsWetBot', 'WellWishesBot',
					  'WikiTextBot', 'WikiSummarizerBot', 'xkcd-Hyphen-bot', 'xkcd_transcriber', 'YoMammaJokebot',
					  'youtubefactsbot', 'YTubeInfoBot', 'sneakpeekbot'}

# Convert set to lowercase
BLACK_LIST_AUTHORS = {author.lower() for author in BLACK_LIST_AUTHORS}


def is_blacklisted(author: str):
	return author.lower() in BLACK_LIST_AUTHORS


def is_remove_or_deleted(line: str):
	if line == "[removed]" or line == "[deleted]":
		return True
	else:
		return False


def extract_comments(comment_tree, current_str, strings):
	for comment_id, comment_details in comment_tree.items():
		new_str = current_str + f'<|context_level|>{comment_details["context_level"]}<|comment|>{comment_details["text"]}'
		if comment_details['replies']:
			extract_comments(comment_details['replies'], new_str, strings)
		else:
			strings.append(new_str + '<|endoftext|>')


async def collect_comments(comment: Comment, context_level=0):
	comment_details = {
		'id': str(comment.id),
		'author': str(comment.author),
		'text': comment.body,
		'replies': {},
		'parent_id': str(comment.parent_id),
		'link_id': str(comment.link_id),
		'context_level': context_level
	}
	comment_replies = comment.replies  # Removed await

	for reply in comment_replies:
		reply_details = await collect_comments(reply, context_level + 1)
		comment_details['replies'][reply.id] = reply_details

	return comment_details


async def get_all_comments(submission):
	try:
		await submission.comments.replace_more(limit=None)
	except asyncprawcore.exceptions.TooManyRequests as e:
		logger.error(e)
		time.sleep(60)
		logger.info("Retrying...")
		await submission.comments.replace_more(limit=None)
	except Exception as e:
		logger.error(e)
		raise e

	root_comments = submission.comments  # Removed await

	comment_tree = {}
	for comment in tqdm(root_comments, total=len(root_comments), desc="Processing comments"):
		author = str(comment.author).lower()
		body = comment.body
		if is_blacklisted(author):
			continue
		if is_remove_or_deleted(body):
			continue
		comment_details = await collect_comments(comment)
		comment_tree[comment.id] = comment_details

	return comment_tree


async def collect_submission_details(submission, processor) -> dict:
	try:
		await submission.load()
		author = str(submission.author).lower()
		url = submission.url.lower()
		has_image = url.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"))
		text = ""
		if has_image:
			async for item in processor.caption_image_from_url(url):
				if item is None:
					continue
				else:
					text = item
		else:
			text = submission.selftext

		if is_blacklisted(author):
			return None

		submission_details = {
			"Id": submission.id,
			"Author": str(submission.author),
			"Subreddit": "r/" + str(submission.subreddit),
			"Title": submission.title,
			"Text": text,
			"Score": submission.score,
			"comments": [],
			"Type": "image" if has_image else "text"
		}
		return submission_details
	except Exception as e:
		logger.error(e)
		return None


async def main():
	image_processor: CaptionProcessor = CaptionProcessor()
	script_dir = Path(__file__).resolve().parent

	cache_path = os.path.join(script_dir, 'cache')
	os.makedirs(cache_path, exist_ok=True)
	cache_db_path = Path(cache_path, "cache.db")
	site_name = os.environ.get("REDDIT_ACCOUNT_SECTION_NAME")
	while True:
		reddit = asyncpraw.Reddit(site_name=site_name)
		with shelve.open(str(cache_db_path)) as db:
			try:
				subreddit = await reddit.subreddit("all")
				pbar = tqdm(subreddit.hot(limit=1000), total=1000, desc="Processing submissions")
				async for submission in subreddit.hot(limit=1000):
					try:
						if submission.id in db:
							continue
						basic_submission = await collect_submission_details(submission, image_processor)
						if basic_submission is None:
							continue
						try:
							comment_tree = await get_all_comments(submission)
							basic_submission["comments"] = comment_tree
						except asyncprawcore.exceptions.TooManyRequests as e:
							logger.error(e)
							time.sleep(60)
							comment_tree = await get_all_comments(submission)
							basic_submission["comments"] = comment_tree
							continue
						except Exception as e:
							logger.error(e)
							continue

						constructed_strings = []
						submission_type: str = basic_submission['Type']
						base_str = f'<|startoftext|><|subreddit|>{basic_submission["Subreddit"]}<|title|>{basic_submission["Title"]}<|{submission_type}|>{basic_submission["Text"]}'
						extract_comments(basic_submission['comments'], base_str, constructed_strings)

						data_path = os.path.join(script_dir, "data")
						os.makedirs(data_path, exist_ok=True)
						filename = Path(data_path, f"{submission.id}.txt")
						filename.touch(exist_ok=True)

						append_blob_path = Path(os.path.join(script_dir, "appendBlob.jsonl"))
						append_blob_path.touch(exist_ok=True)

						with append_blob_path.open("ab") as append_blob:
							blob_string = json.dumps(basic_submission)
							encoded = blob_string.encode('unicode_escape')
							append_blob.write(encoded + b'\n')

						with filename.open("wb") as f:
							for item in constructed_strings:
								encoded = item.encode('unicode_escape')
								f.write(encoded)
								f.write(b'\n')
						db[submission.id] = True
						logging.info(f"Successfully processed and saved submission {submission.id}")
					finally:
						await asyncio.sleep(5)
						pbar.update(1)
			except Exception as e:
				logger.error(f"An error occurred: {e}", exc_info=True)
			finally:
				await reddit.close()

		await asyncio.sleep(60)