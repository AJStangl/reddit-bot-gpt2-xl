from dataclasses import dataclass, asdict


@dataclass
class RedditComment:
    text: str
    image: str
    responding_bot: str
    subreddit: str
    reply_id: str
    type: str
    title: str

    def to_dict(self):
        return asdict(self)
