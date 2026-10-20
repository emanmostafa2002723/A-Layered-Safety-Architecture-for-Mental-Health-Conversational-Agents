"""Bounded account-owned interface choices, separate from personal memory."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AvatarPreferences(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    character: int = Field(default=2, ge=0, le=5)
    colour: int = Field(default=0, ge=0, le=4)
    accessory: Literal['none', 'star', 'flower'] = 'none'


class InterfacePreferences(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    display_name: str = Field(default='', max_length=32)
    avatar: AvatarPreferences = Field(default_factory=AvatarPreferences)
    theme: Literal['day', 'evening'] = 'day'
    text: Literal['default', 'large', 'larger'] = 'default'
    motion: Literal['auto', 'reduce'] = 'auto'
    quiet: bool = False
    country: str | None = Field(default=None, pattern='^[A-Z]{2}$')
    location_confirmed: bool = False
    bookmarks: list[str] = Field(default_factory=list, max_length=100)

    @field_validator('display_name')
    @classmethod
    def valid_name(cls, value):
        value.encode('utf-8', errors='strict')
        if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
            raise ValueError('invalid_display_name')
        return value.strip()

    @field_validator('bookmarks')
    @classmethod
    def valid_bookmarks(cls, values):
        import re
        if any(not re.fullmatch('[a-z0-9-]{1,60}', item) for item in values):
            raise ValueError('invalid_bookmark')
        return list(dict.fromkeys(values))

    @model_validator(mode='after')
    def confirmed_location_requires_country(self):
        if self.location_confirmed and not self.country:
            raise ValueError('country_required_for_location_confirmation')
        return self
