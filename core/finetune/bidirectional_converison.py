import re


class BidirectionalConversation:
	def string_to_ordered_dict(self, input_str):
		# Initialize an empty list to hold the results
		result_list = []

		# Find all occurrences of keys and their positions in the string
		keys_positions = [(m.start(0), m.end(0), m.group(0)) for m in re.finditer(r'<\|.*?\|>', input_str)]

		# Loop through keys and extract the corresponding content
		for i in range(len(keys_positions)):
			start, end, key = keys_positions[i]

			# Determine where the content for this key ends
			end_content = keys_positions[i + 1][0] if i + 1 < len(keys_positions) else len(input_str)

			# Extract the content
			content = input_str[end:end_content].strip()

			# Remove the <| and |> from the key name
			key = key[2:-2]

			# Add to the list
			result_list.append((key, content))

		return result_list


	def ordered_dict_to_string(self, input_list):
		# Initialize an empty string to hold the result
		result_str = ''

		# Loop through the list and append keys and their values to the string
		for key, value in input_list:
			result_str += f"<|{key}|>{value}"

		return result_str.strip()


if __name__ == '__main__':
	import json
	bidirectional_conversation = BidirectionalConversation()
	with open('/core/finetune/data/16a9uzp.txt') as handle:
		lines = handle.readlines()
		for line in lines:
			if line == "":
				continue
			else:
				line = line.strip()
				thing = bidirectional_conversation.string_to_ordered_dict(line)
				print(json.dumps(thing, indent=4))
				print(bidirectional_conversation.ordered_dict_to_string(thing))
