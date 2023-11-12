import glob
import logging
import os
import re
import time
from io import BytesIO
from typing import Optional

import praw
import requests
import torch
from PIL import Image
from dotenv import load_dotenv
from praw.models import Submission
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageData(object):
	def __init__(self, url: Optional[str] = None, caption: Optional[str] = None):
		self.url: Optional[str] = url
		self.caption: Optional[str] = caption


class BlipQA(object):
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained("D:\\code\\repos\\reddit-bot-gpt2-xl\\core\\finetune\\blip").to(self.device)
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


	def load_demo_image(self, image_url, image_size=384, device="cuda"):
		raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
		w, h = raw_image.size
		transform = transforms.Compose([
			transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		])
		image = transform(raw_image).unsqueeze(0).to(device)
		return image

	def caption_image(self, image_url):
		try:
			image_size = 384
			image = self.load_demo_image(image_url=image_url, image_size=image_size, device=self.device)
			with torch.no_grad():
				input_ids = self.tokenizer(["a picture of"], return_tensors="pt").input_ids.to(self.device)
				output_ids = self.model.generate(image, input_ids)
				caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
				caption_new = caption.replace('[UNK]', '').strip()
				return caption_new
		except Exception as e:
			logger.exception(e)
			return None


class CaptionProcessor(object):
	def __init__(self, device_name: str = "cuda"):
		self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
		self.device_name = device_name
		self.device = torch.device(self.device_name)
		self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)

	def caption_image_from_url_local(self, image_url: str) -> Optional[str]:
		max_attempts = 3
		while max_attempts > 0:
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
				max_attempts -= 1
				time.sleep(1)
				continue
		return None


blip = BlipQA()

def get_all_files() -> list[str]:
	txt_files = glob.glob("D:\\data\\text\\*\\*.txt")
	return txt_files


def update_file_with_type(file_path, submission_type) -> str:
	with open(file_path, 'r') as file:
		content = file.read()
		updated_content = content.replace('<|text|>', submission_type)
		return updated_content


def identify_submission_type(submission: Submission) -> str:
	is_image = any([submission.url.endswith(item) for item in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']])
	if is_image:
		return "<|image|>"
	if hasattr(submission, 'is_gallery') and submission.is_gallery:
		return "<|gallery|>"
	if hasattr(submission, 'is_video') and submission.is_video:
		return "<|video|>"
	if hasattr(submission, 'is_self') and submission.is_self:
		return "<|text|>"
	if hasattr(submission, 'selftext') and submission.selftext:
		return "<|text|>"
	if hasattr(submission, 'crosspost_parent_list'):
		return "<|crosspost|>"
	else:
		return "<|link|>"


def file_has_been_processed(file_name: str) -> str:
	dest_listing = os.listdir("D:\\code\\repos\\reddit-bot-gpt2-xl\\core\\finetune\\data\\")
	return file_name in dest_listing


def get_gallery_images(submission):
	if hasattr(submission, 'is_gallery') and submission.is_gallery:
		media_metadata = submission.media_metadata
		images = []

		if media_metadata is None:
			return images

		if media_metadata.items() is None:
			return images

		for item_id, item_data in media_metadata.items():
			image_url = item_data['s']['u']
			images.append(ImageData(url=image_url, caption=""))

		return images


def caption_images(images: list[ImageData]) -> list[ImageData]:
	captions = []
	for image in images:
		caption = blip.caption_image(image.url)
		image.caption = caption
		if caption is not None:
			captions.append(caption)
	return captions


def update_content(content: str, submission_class: str, submission: Submission) -> str:
	try:
		tag_dict = parse_content_to_dict(content)
		submission_class = submission_class.replace("<|", "").replace("|>", "")
		# we have the tag in our data
		if submission_class in tag_dict:
			caption_or_text_value = tag_dict.get(submission_class)
			# we are missing the caption or text and need to get it
			if caption_or_text_value is None or caption_or_text_value == "":
				if submission_class == "gallery":
					captions = handle_gallery(submission)
					all_captions = [item for item in captions]
					joined_captions = ",".join(all_captions).strip()
					content_new = content.replace(f"<|{submission_class}|>", f"<|{submission_class}|>{joined_captions}")
					return content_new
				if submission_class == "image":
					caption = blip.caption_image(submission.url)
					content_new = content.replace(f"<|{submission_class}|>", f"<|{submission_class}|>{caption}")
					return content_new
				if submission_class == "link":
					content_new = content.replace(f"<|{submission_class}|>", f"<|{submission_class}|>{submission.url}")
					return content_new
				if submission_class == "text":
					return content
				if submission_class == "crosspost":
					parent_permalink = submission.crosspost_parent_list[0].get('permalink', '')
					parent_url = f"https://www.reddit.com{parent_permalink}"
					content_new = content.replace(f"<|{submission_class}|>", f"<|{submission_class}|>{parent_url}")
					return content_new
				if submission_class == "video":
					thumbnail = submission.thumbnail
					caption = blip.caption_image(thumbnail)
					content_new = content.replace(f"<|{submission_class}|>", f"<|{submission_class}|>{caption}")
					return content_new
				else:
					return content
			# it's already been processed so there's nothing to do
			else:
				return content
		else:
			return content
	except Exception as e:
		logger.exception(e)
		return None


def handle_gallery(submission: Submission) -> str:
	images = get_gallery_images(submission)
	captions = caption_images(images)
	return captions


def parse_content_to_dict(content: str) -> dict:
	regex_pattern = r'<\|([^|]+)\|>([^<]*)'
	matches = re.findall(regex_pattern, content)
	tag_dict = {tag: value for tag, value in matches}
	return tag_dict


if __name__ == '__main__':
	reddit = praw.Reddit(site_name="PoetBotGPT")
	data_path = "D:\\data\\text\\"
	out_path = ""
	all_files_to_process = get_all_files()
	for file in tqdm(all_files_to_process, total=len(all_files_to_process), desc="Processing"):
		try:
			file_name = os.path.basename(file)
			if file_name == "comment_data.txt":
				continue
			been_processed = file_has_been_processed(file_name)
			if been_processed:
				continue
			submission_id = file_name.split(".")[0]
			submission = praw.reddit.Submission(id=submission_id, reddit=reddit)
			submission_class = identify_submission_type(submission)
			content = update_file_with_type(file, submission_class)
			if content == '':
				continue
			updated_content = update_content(content=content, submission_class=submission_class, submission=submission)
			if updated_content == '' or updated_content is None:
				continue

			splits = updated_content.split("<|endoftext|>")
			if splits[0] == "":
				continue
			out_file = f"D:\\code\\repos\\reddit-bot-gpt2-xl\\core\\finetune\\data\\{file_name}"
			with open(out_file, 'wb') as out:
				for line in splits:
					foo = line.split("<|startoftext|>")[-1]
					bar = f"<|startoftext|>{foo}<|endoftext|>"
					encoded = bar.encode('unicode_escape')
					out.write(encoded)
					out.write(b'\n')
		except Exception as e:
			print(f"{e} - {file}")
			continue

