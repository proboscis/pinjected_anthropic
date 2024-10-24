import asyncio
import base64
import io
from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Callable, Awaitable

import PIL
import pandas as pd
from PIL import Image
from anthropic import AsyncAnthropic, Stream
from anthropic import RateLimitError, InternalServerError
from anthropic.types import Message, MessageStartEvent, ContentBlockStartEvent, ContentBlockDeltaEvent
from pinjected import *


@instance
async def anthropic_client(anthropic_api_key):
    return AsyncAnthropic(api_key=anthropic_api_key)


IMAGE_FORMAT = Literal['jpeg', 'png']


@injected
async def a_anthropic_llm(
        anthropic_client,
        /,
        messages: list[dict],
        max_tokens=1024,
        # model="claude-3-opus-20240229"
        model="claude-3-5-sonnet-20240620"
) -> Message:
    msg = await anthropic_client.messages.create(
        max_tokens=max_tokens,
        model=model,
        messages=messages
    )
    return msg


def image_to_base64(image: PIL.Image.Image, fmt: IMAGE_FORMAT) -> str:
    assert isinstance(image, PIL.Image.Image), f"image is not an instance of PIL.Image.Image: {image}"
    bytes_io = io.BytesIO()
    image.save(bytes_io, format=fmt)
    bytes_io.seek(0)
    data = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
    assert data, "data is empty"
    return data


"""


import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ],
        }
    ],
)
print(message)
"""


@dataclass
class UsageEntry:
    timestamp: pd.Timestamp
    tokens: int

    class Config:
        arbitrary_types_allowed = True


@dataclass
class RateLimitManager:
    max_tokens: int
    max_calls: int
    duration: pd.Timedelta
    lock: Lock = asyncio.Lock()
    call_history: list[UsageEntry] = field(default_factory=list)

    async def acquire(self, approx_tokens):
        if await self.ready(approx_tokens):
            pass
        else:
            # wait for some time or condition, but who checks the condition?
            # a distinct loop, or loop here?
            # 1. check if we need to wait
            # 2. check if someone else is waiting with loop
            # 3. if not use looping to wait
            # Currently, everyone waits with loops
            while not await self.ready(approx_tokens):
                await asyncio.sleep(1)

    async def ready(self, token):
        async with self.lock:
            remaining = await self._remaining_tokens()
            is_ready = remaining >= token and len(self.call_history) < self.max_calls
            if is_ready:
                self.call_history.append(UsageEntry(pd.Timestamp.now(), token))
            return is_ready

    async def _remaining_tokens(self):
        return self.max_tokens - await self._current_usage()

    async def _current_usage(self):
        t = pd.Timestamp.now()
        self.call_history = [e for e in self.call_history if e.timestamp > t - self.duration]
        return sum(e.tokens for e in self.call_history)


@dataclass
class AnthropicRateLimitController:
    manager_factory: Callable[[str], Awaitable[RateLimitManager]]
    managers: dict[str, RateLimitManager] = field(default_factory=dict)
    lock: Lock = asyncio.Lock()

    async def get_manager(self, key):
        async with self.lock:
            if key not in self.managers:
                self.managers[key] = await self.manager_factory(key)
            return self.managers[key]


@instance
async def anthropic_rate_limit_controller():
    async def factory(key: str):
        if 'sonnet' in key and '3_5' in key:
            return RateLimitManager(
                max_tokens=40000,
                max_calls=50,
                duration=pd.Timedelta(minutes=1),
            )
        elif 'opus' in key and '3' in key:
            return RateLimitManager(
                max_tokens=20000,
                max_calls=50,
                duration=pd.Timedelta(minutes=1),
            )
        elif 'sonnet' in key and '3' in key:
            return RateLimitManager(
                max_tokens=40000,
                max_calls=50,
                duration=pd.Timedelta(minutes=1),
            )
        elif 'haiku' in key and '3' in key:
            return RateLimitManager(
                max_tokens=50000,
                max_calls=50,
                duration=pd.Timedelta(minutes=1),
            )
        else:
            return RateLimitManager(
                max_tokens=20000,
                max_calls=50,
                duration=pd.Timedelta(minutes=1),
            )

    return AnthropicRateLimitController(factory)


def count_image_token(img: PIL.Image.Image):
    w, h = img.size
    tokens = (w * h) / 750
    return tokens



@injected
async def a_vision_llm__anthropic(
        anthropic_client: AsyncAnthropic,
        image_to_base64,
        logger,
        anthropic_rate_limit_controller: AnthropicRateLimitController,
        /,
        text: str,
        images: list[PIL.Image.Image] = None,
        model="claude-3-opus-20240229",
        max_tokens: int = 2048,
        img_format: IMAGE_FORMAT = 'jpeg'
) -> str:
    img_blocks = []
    if images is not None:
        for img in images:
            if img_format == 'jpeg':
                img = img.convert('RGB')
            block = {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': f"image/{img_format}",
                    'data': image_to_base64(img, img_format),
                }
            }
            img_blocks.append(block)

    expected_text_tokens = await anthropic_client.count_tokens(text)
    expected_image_tokens = sum([count_image_token(img) for img in images]) if images is not None else 0

    async def attempt():
        msg = await anthropic_client.messages.create(
            model=model,
            messages=[
                {
                    'content': [
                        *img_blocks,
                        {
                            'type': 'text',
                            'text': text
                        },
                    ],
                    'role': "user"
                },
            ],
            max_tokens=max_tokens
        )
        msg: Message
        return msg
    manager:RateLimitManager = await anthropic_rate_limit_controller.get_manager(model)
    from anthropic import APIConnectionError
    while True:
        try:
            await manager.acquire(expected_text_tokens + expected_image_tokens)
            resp = await attempt()
            return resp.content[-1].text
        except RateLimitError as rle:
            logger.warning(f"Rate limit error for model {model}, waiting for {5} seconds")
            await asyncio.sleep(5)
        except InternalServerError as ise:
            logger.warning(f"Rate limit error for model {model}, waiting for {5} seconds")
            await asyncio.sleep(10)
        except APIConnectionError as ace:
            logger.warning(f"API connection error for model {model}, waiting for {5} seconds")
            await asyncio.sleep(10)

@injected
async def a_anthropic_llm_stream(
        anthropic_client,
        /,
        messages: list[dict],
        max_tokens=1024,
        model="claude-3-opus-20240229"
) -> Stream:
    msg = await anthropic_client.messages.create(
        max_tokens=max_tokens,
        model=model,
        messages=messages,
        stream=True
    )
    async for item in msg:
        match item:
            case MessageStartEvent():
                pass
            case ContentBlockStartEvent():
                pass
            case ContentBlockDeltaEvent() as cbde:
                yield cbde.delta.text


test_run_opus: Injected = a_anthropic_llm(
    messages=[
        {
            "content": "What is the meaning of life?",
            "role": "user"
        }
    ],
)

test_a_vision_llm: IProxy = a_vision_llm__anthropic(
    text="What is the meaning of life?",
    images=[],
)
sample_image = injected(Image.open)("test_image/test1.jpg")
test_to_base64: IProxy = injected(image_to_base64)(sample_image, 'jpeg')
test_a_vision_llm_with_image: IProxy = a_vision_llm__anthropic(
    text="What do you see in this image?",
    images=Injected.list(
        injected(Image.open)("test_image/test1.jpg")
    ),
)


@instance
async def test_run_opus_stream(a_anthropic_llm_stream):
    stream = a_anthropic_llm_stream(
        messages=[
            {
                "content": "What is the meaning of life?",
                "role": "user"
            }
        ],
    )
    async for msg in stream:
        print(msg)


__meta_design__ = instances(

)
