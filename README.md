# reddit-bot-gpt2-xl

This repo can be used to quickly collect fine-tuned data, create the appropriate output file, train on collab, and run
the bot on a local machine.

## Setup

## Configuration

There are two sets of configuration. 1 is a `praw.ini` file that must be placed in the root directory. The other is a
`.env` file that must be placed in the root directory. This guide assumes you have read the praw docs and know how to
set up a praw.ini file. The .env requires explanation:

| Variable Name               | Description                                                                                                                                                                                                |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| REDDIT_ACCOUNT_SECTION_NAME | This is the name of the section in the praw.ini file. Keep it simple and have the bot name match the section name, technically this is used for polling and not for posting/commenting                     |
| SUBREDDIT_TO_MONITOR        | This is the sub-reddit to monitor for comments you can use multiple sub reddits by separating them with a +, for example: CoopAndPabloArtHouse+CoopAndPabloPlayhouse                                       |
| REDDIT_BOTS_TO_REPLY        | This is a list of bots accounts (reddit accounts that are associated with the praw.ini file) that will be replied to. This is a comma separated list and MUST match the section name in the praw.ini file. |
| MODEL_PATH                  | This is the path to the model. This is the path to the directory that contains the `pytorch_model.bin` file.                                                                                               |


### Install Dependencies

First, this implementation will require torch, or you will need to fork it and figure out how to load the model. This
technically will work with all GPT2 variants, basic, medium, large, and XL.

So: Before you start, it is recommended to manually install torch, torchvision, and torchaudio. This is because the
requirements file is a pain in the ass and I am lazy.

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then, install the requirements file:

```
pip install -r requirements.txt
```

### Start Collecting Data

From the project directory run the following command and execute the collection routine: (this assumes you have
activated the venv)

```
python reddit-collect-data.py
```

The script goes through `hot` for `all` and goes through the last 1000 hot posts. This can't be modified by
configuration at this time.

In the folder:
`reddit-bot-gpt2-xl\core\finetune\data` there will be a file per submission with the following naming convention
{submission_id}.txt

In addition to an `appendBlob.jsonl` that can be used that represents the json structure used to generate the training
line.

The text files will contain the following pattern per line:

```text
<|startoftext|><|subreddit|>r/{Subreddit}<|title|>{SubmissionTitle}<|text|>{SubmissionTitleOfBlipCaption}<|context_level|>{PositionInCommentTree}<|comment|>{CommentBody}<|context_level|>{PositionInCommentTree}<|comment|>{CommentBody}<|endoftext|>
```

Or otherwise put, it will specify the subreddit, submission title, submission body, and then the comment tree. The
primary and complex job og the script is to properly assemble this string.

Next run the following script found in the project directory:

```
python reddit-create-training.py
```

will output a file called `raw_training_{timestamp}.txt` which is to be used for training in collab.

### Training

Go to collab and upload the notebook. You will need to use an A100 so unless you have a paid account, you will need to
get one or this will not work.

Take the file and place it in the `working` directory of your collab instance. Ensure you specify the `out_model`. The
training for 500,000 lines takes about 10 hours on an A100. Normally, collab will disconnect you after 4-6 hours, so
ensure you are checking in on it and that the checkpoint is being saved. The script limits the max number of checkpoints
to 3 and will delete the oldest one. This is to prevent the disk from filling up.

Once complete download your model and unzip it. Normally, collab will do something stupid like save two zipped files:

```text
out_model.zip
pytorch_model_001.bin.zip
```

The pytorch_model_001.bin.zip is the one you want. Unzip it and rename it to `pytorch_model.bin`. Then, place it in the
other directory. This is the directory that will be set the .evn file.

```text
MODEL_PATH=C:\SOME\PATH\TO\MODEL
```

### Running the Bot

This assumes you have a praw.ini and have read praws docs. If not, you will need to do that first.

Ensure your .env and .praw are set and it run! It will run forever, so you will need to kill it manually.
