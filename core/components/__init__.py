def data(caption):
	return {
		"enable_hr": True,
		"denoising_strength": .1,
		"firstphase_width": 0,
		"firstphase_height": 0,
		"hr_scale": 2,
		"hr_upscaler": "Lanczos",
		"hr_second_pass_steps": 20,
		"hr_resize_x": 1024,
		"hr_resize_y": 1024,
		"hr_sampler_name": "",
		"hr_prompt": f"{caption}",
		"hr_negative_prompt": "",
		"prompt": f"{caption}",
		"styles": [""],
		"seed": -1,
		"subseed": -1,
		"subseed_strength": 0,
		"seed_resize_from_h": -1,
		"seed_resize_from_w": -1,
		"sampler_name": "DDIM",
		"batch_size": 1,
		"n_iter": 1,
		"steps": 20,
		"cfg_scale": 7,
		"width": 512,
		"height": 512,
		"restore_faces": True,
		"tiling": False,
		"do_not_save_samples": False,
		"do_not_save_grid": False,
		"negative_prompt": "",
		"eta": 0,
		"s_min_uncond": 0,
		"s_churn": 0,
		"s_tmax": 0,
		"s_tmin": 0,
		"s_noise": 1,
		"override_settings": {},
		"override_settings_restore_afterwards": True,
		"script_args": [],
		"sampler_index": "DDIM",
		"script_name": "",
		"send_images": True,
		"save_images": False,
		"alwayson_scripts": {}
	}