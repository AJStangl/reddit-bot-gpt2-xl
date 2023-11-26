import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import time

import pandas
import praw
import requests
from PIL import Image
from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from praw.models import Submission
from dataclasses import dataclass
from typing import Dict
from core.components.text.services.image_generation import Runner, ImageGenerationResult

load_dotenv()
logging.getLogger("azure").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMapper:
	def __init__(self):
		self.caption_lookup: dict = json.loads(open('data/caption-lookup.json', 'r').read())
		self.lora_api_response: dict = json.loads(open('data/lora-api-response.json', 'r').read())
		self.model_type_negatives: list[dict] = pandas.read_csv('data/model-type-negatives.tsv', sep='\t').to_dict(orient='records')
		self.negative_prompts: dict = json.loads(open('data/negative-prompts.json', 'r').read())
		self.caption_generator: list = json.loads(open('data/captions.json', 'r', encoding='utf=8').read())


class UtilityFunctions:
	def __init__(self):
		self.data_mapper = DataMapper()
		self.image_bot = ImageBot()
		self.reddit_poster = RedditPoster()

	@staticmethod
	def mask_social_text(text):
		masked_text = re.sub(r'#\w+', '', text)
		masked_text = re.sub(r'@\w+', '', masked_text)
		masked_text = masked_text.strip()
		if len(masked_text) == 0:
			masked_text = "Untitled"
		return masked_text

	def send_image_to_reddit_and_cloud(self, data: ImageGenerationResult):
		try:
			title = data.title
			subject = data.subject
			base_caption = data.caption
			lora = data.subject
			negative_prompt = data.negative_prompt
		except Exception as e:
			logger.exception(e)
			return None

		info_string = f"""
			==================================================
			## Submission Information:

			Title: {title}

			Subject: {subject}

			## Model Information:

			Lora Weight(s): {lora}

			## Prompt:
			==================================================

			{base_caption}

			{negative_prompt}

			==================================================
"""
		try:
			images = self.image_bot.write_image_result_to_remote_and_local(image_generation_result=data, info_string=info_string)
			if len(images) == 0:
				return None
			self.reddit_poster.create_submission(model_name=data.subject, title=title, images=images, info_string=info_string)
			return None
		except Exception as e:
			logger.exception(e)
			return None

	def get_caption_for_subject_name(self, lora_subject_name):
		try:
			if lora_subject_name.__contains__("Sara"):
				lora_subject_name = "sarameikasai"
			captions = self.data_mapper.caption_generator
			random_title_object = [item for item in captions if lora_subject_name in item][0]
			last_random = random.choice(random_title_object[lora_subject_name])
			title = last_random['title']
			caption = random.choice(last_random['captions'])
			return {
				'caption': caption,
				'title': title
			}
		except Exception as e:
			logger.exception(e)
			return None

	@staticmethod
	def handle_special_subject_caption(subject, title, caption):
		if subject in ["celebrities", "gentlemanboners", "CityPorn", "PrettyGirls"]:
			return f"{title}, {caption}"
		else:
			return caption


class AzureCloudStorage:

	@staticmethod
	def write_image_to_cloud(final_image_path: str):

		blob_service_client = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])

		blob_client = blob_service_client.get_blob_client(container="images", blob=os.path.basename(final_image_path))

		try:
			with open(final_image_path, "rb") as f:
				image_data = f.read()
				blob_client.upload_blob(image_data, overwrite=True)
				final_remote_path = f"https://ajdevreddit.blob.core.windows.net/images/{os.path.basename(final_image_path)}"
				return final_remote_path

		except Exception as e:
			logger.exception(e)
			raise Exception(e)

		finally:
			blob_client.close()
			blob_service_client.close()

	@staticmethod
	def write_output_to_table_storage_row(subject_key, iteration, title, caption, style, final_remote_path):
		import datetime
		inverted_ticks = str(int((datetime.datetime.max - datetime.datetime.utcnow()).total_seconds()))
		entity = {
			"PartitionKey": inverted_ticks + "-" + subject_key,
			"RowKey": inverted_ticks + "-" + subject_key + "-" + str(iteration),
			"TimeStamp": datetime.datetime.now(),
			"prompt": caption,
			"title": title,
			"style": style,
			"path": final_remote_path
		}

		service_client: TableServiceClient = TableServiceClient(endpoint=os.environ["AZURE_TABLE_ENDPOINT"],
																credential=AzureNamedKeyCredential(
																	os.environ["AZURE_ACCOUNT_NAME"],
																	os.environ["AZURE_ACCOUNT_KEY"]))
		table_client: TableClient = service_client.get_table_client("generations")
		try:
			table_client.upsert_entity(entity=entity)
		except Exception as e:
			logger.exception(f"{e}")
			table_client.close()
			service_client.close()


class RedditPoster:
	def __init__(self):
		self.reddit = praw.Reddit(site_name=os.environ.get('IMAGE_BOT_ACCOUNT'))
		self.flair_map = json.loads(open('data/flair-map.json', 'r').read())
		self.flair_map = {k.lower(): v for k, v in self.flair_map.items()}
		self.last_submission_id = None

	def create_submission(self, model_name: str, title: str, images: dict, info_string: str):
		try:
			target_sub = "CoopAndPabloArtHouse"

			sub = self.reddit.subreddit(target_sub)

			latest_submission = sub.new(limit=1)

			latest_submission = latest_submission.__next__()

			self.last_submission_id = latest_submission.id

			flair_id = self.flair_map.get(model_name.lower())

			if len(title) == 0:
				title = "Untitled"
			if len(title) > 179:
				title = title[0:179]

			if len(images) > 1:
				logger.info(f":: Submitting Gallery: {title}")
				sub.submit_gallery(title=f"{title}", images=images, flair_id=flair_id)
			else:
				sub.submit_image(title=f"{title}", image_path=images[0]['image_path'], flair_id=flair_id)

			while True:
				submission: Submission = sub.new(limit=1).__next__()
				if submission.id == self.last_submission_id:
					time.sleep(5)
					continue
				else:
					self.last_submission_id = submission.id
					break
			submission.mod.approve()
			submission.reply(body=info_string[0:9999])

		except Exception as e:
			logger.exception(e)


class ImageBot:
	def __init__(self):
		self.loras = None
		self.loras = self.get_loras()
		self.azure_cloud_storage = AzureCloudStorage()
		self.text_lock_path = os.path.join(os.environ.get("LOCK_PATH"), "text.lock")
		self.image_lock_path = os.path.join(os.environ.get("LOCK_PATH"), "sd.lock")

	def create_lock(self):
		try:
			logger.debug("Creating lock")
			with open(self.image_lock_path, "wb") as handle:
				handle.write(b"")
		except Exception as e:
			logger.error(f"An error occurred while creating temp lock: {e}")

	def clear_lock(self):
		try:
			logger.debug("Clearing lock")
			if os.path.exists(self.image_lock_path):
				os.remove(self.image_lock_path)
			else:
				logger.warning(f"Lock file {self.image_lock_path} does not exist.")
		except Exception as e:
			logger.error(f"An error occurred while deleting text lock: {e}")

	def write_image_result_to_remote_and_local(self, image_generation_result: ImageGenerationResult, info_string: str):
		try:
			data = []
			subject = image_generation_result.subject
			caption = image_generation_result.caption

			out_path = f"D:\\code\\repos\\reddit-bot-gpt2-xl\\output\\{image_generation_result.subject}\\"
			os.makedirs(out_path, exist_ok=True)
			for i, _ in enumerate(image_generation_result.image):
				image: Image = _[0]
				save_path = os.path.join(out_path, f'{i}-{image_generation_result.image_name}')
				image.save(save_path)
				if len(image_generation_result.caption) > 180:
					caption = caption[:177] + "..."
				data.append({
					'image_path': save_path,
					'caption': caption,
				})
				self.azure_cloud_storage.write_image_to_cloud(save_path)
				AzureCloudStorage.write_output_to_table_storage_row(
					subject_key=image_generation_result.subject,
					iteration=i,
					title=image_generation_result.title,
					caption=image_generation_result.caption,
					style=image_generation_result.subject,
					final_remote_path=f"https://ajdevreddit.blob.core.windows.net/images/{os.path.basename(save_path)}"
				)
			with open(os.path.join(out_path, f'{image_generation_result.image_name}.txt'), 'w', encoding='utf-8') as f:
				f.write(info_string)
			return data

		except Exception as e:
			logger.exception(e)
			return {}

	def get_loras(self):
		if self.loras is None:
			try:
				target_loras = []
				if os.path.exists('data/lora-api-response.json'):
					r_json = json.loads(open('data/lora-api-response.json').read())
				else:
					r = requests.get("http://127.0.0.1:7860/sdapi/v1/loras")
					r_json = r.json()

				for elem in r_json:
					if not elem['path'].endswith('.safetensors'):
						continue
					else:
						target_loras.append(elem)
				self.loras = target_loras
				return self.loras
			except Exception as e:
				logger.exception(e)
				self.get_loras()
		return self.loras


if __name__ == '__main__':
	utility_functions: UtilityFunctions = UtilityFunctions()
	runner: Runner = Runner()
	lock_path = os.environ.get("LOCK_PATH", "")
	text_lock_path = os.path.join(lock_path, "text.lock")
	while True:
		while os.path.exists(text_lock_path):
			continue
		result = runner.run_generation(num_images=4)
		try:
			result.title = utility_functions.mask_social_text(result.title)
			utility_functions.send_image_to_reddit_and_cloud(result)
		except Exception as e:
			print(e)
			continue