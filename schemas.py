import pydantic as _pydantic
from typing import Optional
from typing import List


class _RequestBase(_pydantic.BaseModel):
    resize_width: int = 256
    hand_orientation: str = 'right'


class PalmCreate(_RequestBase):
    encoded_base_img: List[str]
