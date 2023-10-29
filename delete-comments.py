import praw
from tqdm import tqdm
from datetime import datetime, timedelta

if __name__ == '__main__':

	# List of usernames to search for
	usernames_to_delete = ["PlayHouseBot-Gpt2", "KimmieBotGPT", "MeganBotGPT", "GaryBot-GPT2", "LauraBotGPT", "PoetBotGPT", "FunnyGuyGPT"]

	# Subreddit you want to scan
	subreddit_name = "ohbehave"

	# Current time
	current_time = datetime.utcnow()

	# Initialize Reddit API for multiple accounts
	for username in usernames_to_delete:
		reddit = praw.Reddit(site_name=username)
		subreddit = reddit.subreddit(subreddit_name)
		latest_submissions = list(subreddit.new(limit=10))
		# Loop through submissions in the subreddit
		for submission in tqdm(latest_submissions, desc=f"Searching for comments by {username}", total=len(latest_submissions)):
			submission_time = datetime.utcfromtimestamp(submission.created_utc)
			if current_time - submission_time > timedelta(hours=24):

				# Fetch comments
				submission.comments.replace_more(limit=0)
				all_comments = submission.comments.list()

				# Loop through comments, delete if authored by username
				for comment in tqdm(all_comments, desc=f"Deleting comments by {username}", total=len(all_comments)):
					if comment.author and comment.author.name == username:
						print(f"Deleting comment {comment.id} by {username}")
						comment.delete()
