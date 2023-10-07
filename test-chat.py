from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
new_model = "C:\\Users\\AJ Stangl\\Downloads\\gpt-xl-reddit-3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained(new_model)
tokenizer.padding_side = "left"
model = GPT2LMHeadModel.from_pretrained(new_model)
model.to(device)

topic = "SexyTimes"
title = "Let's Talk About Sex"
text = "Today we will be discussing how to give the best head"
context_level = 0
base_prompt = f"<|startoftext|><|subreddit|>r/{topic}<|title|>{title}<|text|>{text}<|context_level|>{context_level}<|comment|>"
while True:
	if context_level > 20:
		context_level = 0
		base_prompt = f"<|startoftext|><|subreddit|>r/{topic}<|title|>{title}<|text|>{text}<|context_level|>{context_level}<|comment|>"

	p = input(f"Enter text: Current Prompt: {base_prompt}")
	if p == "f":
		break

	context_level = context_level + 1
	prompt = base_prompt + p + f"<|context_level|>{context_level}<|comment|>"
	encoding = tokenizer(prompt, padding=False, return_tensors='pt').to(device)
	inputs = encoding['input_ids']
	attention_mask = encoding['attention_mask']
	config = {
		'inputs': inputs,
		'attention_mask': attention_mask,
		'max_length': 512,
		'repetition_penalty': 1.1,
		'num_return_sequences': 1,
		'temperature': 1.2,
		'top_k': 50,
		'top_p': 0.95,
		'do_sample': True,
	}

	with (torch.no_grad()):
		generate = True
		while generate:
			for i, _ in enumerate(model.generate(**config)):
				current_prompt = \
					f"""
                    ++++ current prompt ++++
                    {p}
                """
				print(current_prompt)
				generated_texts = tokenizer.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=True)
				filtered_response = generated_texts.replace(prompt, "").replace("<|endoftext|>", "")
				if filtered_response.__contains__("<|context_level|>"):
					filtered_response = filtered_response.split("<|context_level|>")[0]
				else:
					total_words = len(filtered_response.split(" "))
					if total_words == 0 or total_words == 1:
						generate = False
						context_level = 100
						continue
					context_level = context_level + 1
					next_prompt = prompt + filtered_response + f"<|context_level|>{context_level}<|comment|>"
					result = f"""
                    ++++ RESPONSE ++++
                    {filtered_response}
                    """
					base_prompt = next_prompt
					print(result)
					generate = False