import logging
import os
import time
from io import BytesIO
from tqdm import tqdm

import praw
import prawcore
import requests
import torch
from PIL import Image
from praw.models import Comment
from dotenv import load_dotenv
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

	def caption_image_from_url_local(self, image_url: str):
		try:
			with requests.session() as session:
				with session.get(image_url) as response:
					if response.status_code != 200:
						return None
					content = response.content
					image = Image.open(BytesIO(content))
					try:
						inputs = self.processor(images=image, return_tensors="pt").to(self.device)
						out = self.model.generate(**inputs, max_new_tokens=77, num_return_sequences=1, do_sample=True)
						return self.processor.decode(out[0], skip_special_tokens=True)
					except Exception as e:
						logger.exception(e)
						return None
					finally:
						image.close()
		except Exception as e:
			logger.exception(e)
			return None


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


def construct_context_string(comment: Comment) -> str:
	things = []
	current_comment = comment
	counter = 0
	try:
		while not isinstance(current_comment, praw.models.Submission):
			try:
				thing = {
					"text": "",
					"counter": 0
				}
				if thing is None:
					current_comment = current_comment.parent()
					continue
				thing['counter'] = counter
				thing['text'] = current_comment.body
				things.append(current_comment.body)
				counter += 1
				if counter == 8:
					break
				else:
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

	out += f"<|endoftext|>"
	return out


def process_comment(comment: Comment, reddit, caption_processor) -> str:
	if comment is None:
		return

	submission_id = comment.submission
	submission = reddit.submission(submission_id)
	sub_name = str(submission.subreddit)
	mapped_submission = {
		"subreddit": 'r' + '/' + sub_name,
		"title": submission.title,
		"text": submission.selftext,
		"image": submission.url if submission.url.endswith(".jpg") or submission.url.endswith(".png") or submission.url.endswith(".jpeg") else None,
	}

	image_link = mapped_submission.get("image")
	if image_link is not None:
		caption = caption_processor.caption_image_from_url_local(image_link)
		if caption is not None:
			constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|image|>{caption}"
		else:
			constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"
	else:
		constructed_string = f"<|startoftext|><|subreddit|>{mapped_submission['subreddit']}<|title|>{mapped_submission['title']}<|text|>{mapped_submission['text']}"

	constructed_string += construct_context_string(comment)

	return constructed_string


def main():
	caption_processor: CaptionProcessor = CaptionProcessor()
	reddit = praw.Reddit(site_name=os.environ.get("REDDIT_ACCOUNT_SECTION_NAME"))

	comment_stream = reddit.subreddit("all").stream.comments(pause_after=-1)
	with open("comment_data.txt", "ab") as f, tqdm(desc='comments-processed') as pbar:
		for streamed_comment in comment_stream:
			try:
				if streamed_comment is None:
					continue
				if is_blacklisted(str(streamed_comment.author)):
					continue
				constructed_string = process_comment(streamed_comment, reddit, caption_processor)
				encoded = constructed_string.encode('unicode_escape')
				f.write(encoded)
				f.write(b'\n')
			except Exception as e:
				logger.exception(e)
			finally:
				pbar.update(1)