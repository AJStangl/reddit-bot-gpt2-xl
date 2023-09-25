import base64
import io
import os
import random
import json
import time

import requests
from PIL import Image


from asyncpraw import Reddit

import warnings

from asyncpraw.models import Submission

warnings.filterwarnings("ignore")

import logging
import re

from dotenv import load_dotenv
load_dotenv()


class LoggingExtension(object):
	@staticmethod
	def set_global_logging_level(level=logging.ERROR, prefices=[""]):
		prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
		for name in logging.root.manager.loggerDict:
			if re.match(prefix_re, name):
				logging.getLogger(name).setLevel(level)

	@staticmethod
	def get_logging_format() -> str:
		logging_format = f'%(asctime)s %(threadName)s %(levelname)s %(message)s'
		return logging_format


class RedditHandler:
	def __init__(self):
		self.counter: int = 0
		self.reddit: Reddit = Reddit(site_name=os.environ.get("IMAGE_BOT_ACCOUNT"))
		self.flair_map = {
			"AesPleasingAsianGirls": "3f3db71e-c116-11ed-bc88-4257f93035d0",
			"AmIhotAF": "49b00d00-c116-11ed-80c5-7ef7afdcdf7d",
			"Amicute": "6c02c0aa-c116-11ed-a36b-625bab71eac2",
			"AsianInvasion": "52782e72-c116-11ed-8d42-9226dee3c916",
			"AsianOfficeLady": "80192e8a-c116-11ed-9afc-061002270b1c",
			"CollaredDresses": "4b0844f8-c68c-11ed-8e01-0a0ff85df53d",
			"DLAH": "548c5f8a-c70b-11ed-ab4c-d6ecb5af116d",
			"Dresses": "331f2a78-ccef-11ed-b813-beb8ea0d6477",
			"DressesPorn": "0e69c05e-cf41-11ed-a6d7-c265ef3d634d",
			"HotGirlNextDoor": "e978fd72-d0cc-11ed-802d-922e8d939dd5",
			"Ifyouhadtopickone": "7aedfca4-d676-11ed-9536-6a42b6ad77bd",
			"KoreanHotties": "25d50538-d6f7-11ed-9f0f-6a1b95511d30",
			"PrettyGirls": "35eff210-1e6a-11ee-9c05-12d7f8869ab6",
			"SFWNextDoorGirls": "3f47b988-1e6a-11ee-9d1c-6afa6515589a",
			"SFWRedheads": "4abbe03c-1e6a-11ee-99cc-32c698a681d3",
			"SlitDresses": "517fee4a-1e6a-11ee-a10e-260e8c70fccb",
			"TrueFMK": "588a26ba-1e6a-11ee-b536-ae8ffe60fa52",
			"WomenInLongDresses": "5f98a0e4-1e6a-11ee-91a0-e230e35aa34e",
			"amihot": "665574c0-1e6a-11ee-8bc8-dafcb67b8647",
			"celebrities": "6c5d7db8-1e6a-11ee-9b70-b2618031b564",
			"cougars_and_milfs_sfw": "713d0196-1e6a-11ee-bbc5-4a2bbd93586c",
			"gentlemanboners": "75e31316-1e6a-11ee-b595-8288582d46a0",
			"hotofficegirls": "7b7ff3fc-1e6a-11ee-8d93-322a820bd2a4",
			"prettyasiangirls": "8f23c97e-1e6a-11ee-945e-3e9d0fef66f3",
			"realasians": "960dcaaa-1e6a-11ee-b141-3e70be78c500",
			"selfies": "9cfc95bc-1e6a-11ee-af66-e612f9ba8612",
			"sfwpetite": "a798d198-1e6a-11ee-8f84-be52c90e1902",
			"tightdresses": "b0bb5ffc-1e6a-11ee-b82d-ee6aa8a16365"
		}

	def get_markdown_comment(self, prompt, sub, guidance, num_steps, attention_type):
		body =f"""\
| Prompt   | Model Name | Guidance   | Number Of Inference Steps | Attention Type   |
|:--------:|:----------:|:----------:|:------------------------:|:----------------:|
| {prompt} |   {sub}    | {guidance} |           {num_steps}     |  {attention_type}|
"""
		return body

	async def create_submission(self,
						  title: str,
						  image_output: str,
						  model_name: str,
						  prompt: str,
						  guidance: str,
						  num_steps: str,
						  attention_type: str = "",
						  target_sub: str = "CoopAndPabloArtHouse"):
		try:
			sub = await self.reddit.subreddit(target_sub)
			await sub.load()

			submission: Submission = await sub.submit_image(
				title=f"{title}",
				image_path="temp.png",
				nsfw=False,
				flair_id=self.flair_map.get(model_name))
			logger.info(f"https://www.reddit.com/{submission.id}")
			await submission.mod.approve()
			body = self.get_markdown_comment(prompt, model_name, guidance, num_steps, attention_type)
			await submission.reply(body=body)
			return True

		except Exception as e:
			print(e)
			return False


class ImageBot:
	def __init__(self):
		self.loras = self.get_loras()

	def get_image(self, base_caption: str):

		data = {
			"enable_hr": True,
			"denoising_strength": 0.7,
			"firstphase_width": 0,
			"firstphase_height": 0,
			"hr_scale": 2,
			"hr_upscaler": "Lanczos",
			"hr_second_pass_steps": 20,
			"hr_resize_x": 1024,
			"hr_resize_y": 1024,
			"hr_sampler_name": "",
			"hr_prompt": "",
			"hr_negative_prompt": "",
			"prompt": f"{base_caption}",
			"styles": [""],
			"seed": -1,
			"subseed": -1,
			"subseed_strength": 0,
			"seed_resize_from_h": -1,
			"seed_resize_from_w": -1,
			"sampler_name": "DPM++ 2S a",
			"batch_size": 1,
			"n_iter": 1,
			"steps": 30,
			"cfg_scale": 7,
			"width": 512,
			"height": 512,
			"restore_faces": True,
			"tiling": False,
			"do_not_save_samples": False,
			"do_not_save_grid": False,
			"negative_prompt": "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless",
			"eta": 0,
			"s_min_uncond": 0,
			"s_churn": 0,
			"s_tmax": 0,
			"s_tmin": 0,
			"s_noise": 1,
			"override_settings": {},
			"override_settings_restore_afterwards": True,
			"script_args": [],
			"sampler_index": "DPM++ 2S a",
			"script_name": "",
			"send_images": True,
			"save_images": False,
			"alwayson_scripts": {}
	}
		try:
			_response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", headers={'accept': 'application/json', 'Content-Type': 'application/json'}, json=data)
			r = _response.json()
			for i in r['images']:
				image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
				image.save('temp.png')
				return 'temp.png'
		except Exception as e:
			print(e)
			return None


	def get_loras(self):
		try:
			target_loras = []
			r = requests.get("http://127.0.0.1:7860/sdapi/v1/loras")
			for elem in r.json():
				if elem['name'].__contains__('step'):
					continue
				if elem['name'].__contains__('_converted'):
					continue
				if not elem['path'].endswith('.safetensors'):
					continue
				else:
					target_loras.append(elem)
			return target_loras
		except:
			time.sleep(10)
			return self.get_loras()



async def main_async():
	reddit_handler: RedditHandler = RedditHandler()
	image_bot: ImageBot = ImageBot()

	with open('D:\\code\\repos\\image-generation\\look_up.json', 'r', encoding='utf-8') as f:
		subject_map = json.loads(f.read())

	while True:
		random_lora = random.choice(image_bot.loras)
		subject = random_lora['name'].lower()
		chosen_lora_data = subject_map.get(subject)
		captions = list(set(item) for item in random_lora['metadata']['ss_tag_frequency'].values())[0]
		if chosen_lora_data is None:
			continue
		title = random.choice(chosen_lora_data['titles'])
		base_caption = random.choice(list(captions))
		base_caption += f" <lora:{random_lora['name']}:1>"

		print(f"Title: {title}")
		print(f"Subject: {subject}")
		print(f"Caption: {base_caption}")
		image_path = image_bot.get_image(base_caption=base_caption)
		if image_path is not None:
			await reddit_handler.create_submission(title=title, image_output=image_path, model_name=subject,
												   prompt=base_caption, guidance=0, num_steps=50, attention_type='a111')

if __name__ == '__main__':
	import asyncio
	warnings.filterwarnings("ignore")
	LoggingExtension.set_global_logging_level(logging.FATAL, prefices=['diffusers', 'transformers', 'torch', 'praw', 'azure'])
	logging.basicConfig(level=logging.INFO, format=LoggingExtension.get_logging_format(), datefmt='%Y-%m-%d %H:%M:%S')
	logger = logging.getLogger(__name__)
	asyncio.run(main_async())
