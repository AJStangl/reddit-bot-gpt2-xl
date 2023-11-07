import praw

if __name__ == '__main__':
	bot_name = "KimmieBotGPT"
	reddit = praw.Reddit(site_name=bot_name)
	subreddit = reddit.subreddit("CoopAndPabloPlayHouse")
	image_path = "D:\\code\\repos\\reddit-bot-gpt2-xl\\output\\wallstreetbets\\8b88d7c069feed71d64203a79476034c-2.png"
	subreddit.submit_image(title="Gave out $BABA last week went 2X - NEW YOLO - $DIS $89 Calls / Oct 24 2023 Expiration / ~$58k cost basis - wish me luck regards", image_path=image_path, flair_text="YOLO", flair_id="0513bea8-4f64-11e9-886d-0e2b4fe7300c")

