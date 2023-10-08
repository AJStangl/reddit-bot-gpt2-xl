from enum import Enum


class QueueType(Enum):
	GENERATION = 'generation'
	REPLY = 'reply'
	POST = 'post'
