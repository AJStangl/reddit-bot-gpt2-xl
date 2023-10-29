import hashlib
import json
import logging
import os
import random
from typing import Optional

import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler, AutoencoderKL, \
	StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, \
	StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from dotenv import load_dotenv
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, pipeline, BlipForConditionalGeneration, \
	BertTokenizer

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageGenerationResult:
	def __init__(self, title: str, caption: str, negative_prompt: str, transferred_image: Image, subject: str):
		self.title: str = title
		self.caption: str = caption
		self.negative_prompt: str = negative_prompt
		self.image: Image = transferred_image
		self.subject: str = subject
		self.image_name: str = f"{hashlib.md5(caption.encode()).hexdigest()}.png"


class TitleCaptionPair:
	def __init__(self, title, caption):
		self.title = title
		self.caption = caption


class TextToImage:
	def __init__(self):
		self.stable_diffusion_model_path = os.environ.get("SD_MODEL", "")
		self.stable_diffusion_xl_model_path = os.environ.get("SD_MODEL_XL", "")
		self.stable_diffusion_xl_image_to_image_model_path = os.environ.get("SD_MODEL_XL_IMG_TO_IMG", "")
		self.stable_diffusion_upscale_model_path = os.environ.get("SD_MODEL_UPSCALE", "")
		self.device_name = "cuda"
		self.accelerator = Accelerator()

	def enhance_prompt(self, prompt: str) -> str:
		pipe = pipeline("text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion")
		result = pipe(prompt, max_length=50, do_sample=True, temperature=0.9, top_k=50, top_p=0.95,
					  repetition_penalty=1.0, num_return_sequences=1)
		return result[0]['generated_text']

	def get_title_caption_pair_for_lora(self, lora_name: str) -> TitleCaptionPair:
		logger.info(f":: Finding {lora_name}")
		caption_generator: dict = json.loads(
			open('D:\\code\\repos\\reddit-bot-gpt2-xl\\data\\captions.json', 'r', encoding='utf=8').read())
		if lora_name == "SaraMei":
			lora_name = "sarameikasai"
		if lora_name == "KatieBeth":
			lora_name = "princesskatiebeth"

		filtered_list: list = [d for d in caption_generator if lora_name in d.keys()]
		if len(filtered_list) > 0:
			caption_object: dict = filtered_list[0]
			data: list = caption_object[lora_name]
			random_selection: str = random.choice(data)
			title: str = random_selection['title']
			caption: str = random.choice(random_selection['captions'])
			return TitleCaptionPair(title=title, caption=caption)
		else:
			return None

	def assemble_stable_diffusion_pipeline(self) -> StableDiffusionPipeline:
		vae: AutoencoderKL = AutoencoderKL.from_pretrained(self.stable_diffusion_model_path, subfolder='vae')
		text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(self.stable_diffusion_model_path,
																	subfolder='text_encoder')
		tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(self.stable_diffusion_model_path,
																 subfolder='tokenizer')
		unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(self.stable_diffusion_model_path,
																		  subfolder='unet')
		scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(
			self.stable_diffusion_model_path,
			subfolder="scheduler")
		safety_checker: StableDiffusionSafetyChecker = None
		feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(self.stable_diffusion_model_path,
																				   subfolder='feature_extractor')
		requires_safety_checker: bool = False

		stable_diffusion_pipeline: StableDiffusionPipeline = StableDiffusionPipeline(
			vae=vae,
			text_encoder=text_encoder,
			tokenizer=tokenizer,
			unet=unet,
			scheduler=scheduler,
			feature_extractor=feature_extractor,
			safety_checker=safety_checker,
			requires_safety_checker=requires_safety_checker)
		stable_diffusion_pipeline: StableDiffusionPipeline = stable_diffusion_pipeline.to(torch_device="cuda",
																						  torch_dtype=torch.float16)
		return stable_diffusion_pipeline

	def assemble_stable_diffusion_xl_text_to_image_pipeline(self) -> StableDiffusionXLPipeline:
		pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(self.stable_diffusion_xl_model_path,
																					torch_dtype=torch.float16)
		return pipe

	def assemble_stable_diffusion_upscale_pipeline(self) -> StableDiffusionUpscalePipeline:
		model_id = "stabilityai/sd-x2-latent-upscaler"
		upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
		return upscaler

	def assemble_stable_diffusion_xl_image_to_image_pipeline(self) -> StableDiffusionXLImg2ImgPipeline:
		pipe: StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
			self.stable_diffusion_xl_model_path, torch_dtype=torch.float16)
		return pipe

	def create_image(self, prompt: str, negative_prompt: str, lora_name: Optional[str]) -> dict:
		logger.info(f"Generating Image With StableDiffusionPipeline On Prompt: {prompt}")
		pipe: StableDiffusionPipeline = self.assemble_stable_diffusion_pipeline()
		pipe.to(torch_device="cuda")
		if lora_name is not None:
			lora_path = os.path.join("D:\\code\\repos\\stable-diffusion-webui\\models\\Lora", lora_name)
			pipe.unet.load_attn_procs(lora_path)
		try:

			sd_pipeline_output: StableDiffusionPipelineOutput = pipe(prompt=prompt, negative_prompt=negative_prompt,
																	 guidance_scale=7, num_inference_steps=20,
																	 output_type="latent")
			latents = sd_pipeline_output.images
			with torch.no_grad():
				images = pipe.decode_latents(latents)
				return {
					"images": [pipe.numpy_to_pil(item) for item in images],
					"latents": latents
				}
		except Exception as e:
			logger.exception(e)
			return None

		finally:
			del pipe
			torch.cuda.empty_cache()

	def upscale_original_image(self, image: Image, prompt: str):
		upscaler: StableDiffusionUpscalePipeline = self.assemble_stable_diffusion_upscale_pipeline()
		upscaler.to("cuda")
		generator = torch.manual_seed(33)
		try:
			upscaled_image = upscaler(
				prompt=prompt,
				image=image,
				num_inference_steps=20,
				guidance_scale=0,
				generator=generator
			)
			return upscaled_image.images
		except Exception as e:
			logger.exception(e)
			return None
		finally:
			del upscaler
			torch.cuda.empty_cache()

	def create_image_from_image(self, image: Image, prompt: str) -> Image:
		logger.info(f"Generating Image With StableDiffusionXLImg2ImgPipeline From Image")
		pipe: StableDiffusionXLImg2ImgPipeline = self.assemble_stable_diffusion_xl_image_to_image_pipeline()
		pipe.to(torch_device="cuda")
		try:
			sd_pipeline_output: StableDiffusionXLPipelineOutput = pipe(prompt=prompt, image=image, target_size=(1024, 1024))
			images = sd_pipeline_output.images
			return images
		except Exception as e:
			logger.exception(e)
			return None

		finally:
			del pipe
			torch.cuda.empty_cache()

	def caption_image(self, image: Image):
		def load_image(image, image_size=384, device="cuda") -> Optional[Tensor]:
			try:

				w, h = image.size
				transform = transforms.Compose([
					transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
					transforms.ToTensor(),
					transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
				])
				image = transform(image).unsqueeze(0).to(device)
				return image
			except Exception as e:
				logger.exception(e)
				return None

		def caption_image(image: Image) -> str:
			try:
				image_size = 384
				image = load_image(image=image, image_size=image_size, device=device)
				with torch.no_grad():
					input_ids = tokenizer(["a picture of"], return_tensors="pt").input_ids.to(device)
					output_ids = model.generate(image, input_ids)
					caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
					caption_new = caption.replace('[UNK]', '').strip()
					return caption_new
			except Exception as e:
				logger.exception(e)
				return None

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(
			"D:\\models\\blip-captioning\\blip").to(device)
		tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		try:
			caption = caption_image(image=image)
			return caption
		except Exception as e:
			logger.exception(e)
			return None
		finally:
			del model
			del tokenizer
			torch.cuda.empty_cache()


class Runner:
	@staticmethod
	def run_generation():
		try:
			tti = TextToImage()
			negative_prompt = "3D, Absent limbs, Additional appendages, Additional digits, Additional limbs, Altered appendages, Amputee, Asymmetric, Asymmetric ears, Bad anatomy, Bad ears, Bad eyes, Bad face, Bad proportions, Beard , Broken finger, Broken hand, Broken leg, Broken wrist, Cartoon, Childish , Cloned face, Cloned head, Collapsed eyeshadow, Combined appendages, Conjoined, Copied visage, Corpse, Cripple, Cropped head, Cross-eyed, Depressed, Desiccated, Disconnected limb, Disfigured, Dismembered, Disproportionate, Double face, Duplicated features, Eerie, Elongated throat"
			lora_names = ["sarameikasai", "heytegan", "AesPleasingAsianGirls", "miakhalifa", "TrueFMK", "PrettyGirls", "aesha.patel.110696", "Amicute", "amihot"]
			random.shuffle(lora_names)
			for lora_name in lora_names:
				title_caption_pair: TitleCaptionPair = tti.get_title_caption_pair_for_lora(lora_name=lora_name)
				enhanced_title_caption_pair: TitleCaptionPair = TitleCaptionPair(title=title_caption_pair.title, caption=tti.enhance_prompt(prompt=title_caption_pair.caption))
				data = tti.create_image(prompt=enhanced_title_caption_pair.caption, negative_prompt=negative_prompt, lora_name=lora_name)

				for i, image in enumerate(data['images']):
					upscaled_images = tti.upscale_original_image(image=data['latents'][i], prompt=enhanced_title_caption_pair.caption)
					for upscaled_image in upscaled_images:
						generic_caption: str = tti.caption_image(image=upscaled_image)
						enhanced_caption: str = tti.enhance_prompt(prompt=generic_caption)
						create_image_from_image = tti.create_image_from_image(image=upscaled_image, prompt=enhanced_caption)
						for transferred_image in create_image_from_image:
							image_generation_result: ImageGenerationResult = ImageGenerationResult(title=enhanced_title_caption_pair.title, caption=enhanced_title_caption_pair.caption, negative_prompt=negative_prompt, transferred_image=transferred_image, subject=lora_name)
							return image_generation_result
		except Exception as e:
			logger.exception(e)
			return None

	@staticmethod
	def run_generation_deterministic(title: str, prompt: str) -> ImageGenerationResult:
		try:
			tti = TextToImage()
			negative_prompt = "3D, Absent limbs, Additional appendages, Additional digits, Additional limbs, Altered appendages, Amputee, Asymmetric, Asymmetric ears, Bad anatomy, Bad ears, Bad eyes, Bad face, Bad proportions, Beard , Broken finger, Broken hand, Broken leg, Broken wrist, Cartoon, Childish , Cloned face, Cloned head, Collapsed eyeshadow, Combined appendages, Conjoined, Copied visage, Corpse, Cripple, Cropped head, Cross-eyed, Depressed, Desiccated, Disconnected limb, Disfigured, Dismembered, Disproportionate, Double face, Duplicated features, Eerie, Elongated throat"
			enhanced_prompt = tti.enhance_prompt(prompt=prompt)
			enhanced_title_caption_pair: TitleCaptionPair = TitleCaptionPair(title=title, caption=enhanced_prompt)
			data = tti.create_image(prompt=enhanced_title_caption_pair.caption, negative_prompt=negative_prompt, lora_name=None)
			for i, image in enumerate(data['images']):
				upscaled_images = tti.upscale_original_image(image=data['latents'][i], prompt=enhanced_title_caption_pair.caption)
				for upscaled_image in upscaled_images:
					generic_caption: str = tti.caption_image(image=upscaled_image)
					enhanced_caption: str = tti.enhance_prompt(prompt=generic_caption)
					create_image_from_image = tti.create_image_from_image(image=upscaled_image, prompt=enhanced_caption)
					for transferred_image in create_image_from_image:
						image_generation_result: ImageGenerationResult = ImageGenerationResult(title=enhanced_title_caption_pair.title, caption=enhanced_title_caption_pair.caption, negative_prompt=negative_prompt, transferred_image=transferred_image, subject="None")
						return image_generation_result
		except Exception as e:
			logger.exception(e)
			return None


if __name__ == '__main__':
	runner: Runner = Runner()
	while True:
		result = runner.run_generation_deterministic("I cant believe this", "A Picture of a Tweet with an angry white person")
		result.image.show()
