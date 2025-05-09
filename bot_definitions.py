# bot_definitions.py
"""Definitions of different bot types for the bot registry."""

from bot_registry import BotRegistry, BotType
from bot_runner_helpers import (
    create_call_transfer_settings,
    create_simple_dialin_settings,
    create_simple_dialout_settings,
)

# Create and configure the bot registry
bot_registry = BotRegistry()

# Helper function for silence detection settings
def create_silence_detection_settings(body):
    """Create silence detection settings based on configuration."""
    # Default silence detection settings
    silence_detection_settings = {
        "testInPrebuilt": False,
        "silenceThreshold": 10.0,  # seconds
        "maxUnansweredPrompts": 3
    }
    
    # If silence_detection already exists, merge the defaults with the existing settings
    if "silence_detection" in body:
        existing_settings = body["silence_detection"]
        # Update defaults with existing settings (existing values will override defaults)
        for key, value in existing_settings.items():
            silence_detection_settings[key] = value
    
    return silence_detection_settings

# Register bot types
bot_registry.register(
    BotType(
        name="call_transfer",
        settings_creator=create_call_transfer_settings,
        required_settings=["dialin_settings"],
        incompatible_with=["simple_dialin", "simple_dialout", "voicemail_detection", "silence_detection"],
        auto_add_settings={"dialin_settings": {}},
    )
)

bot_registry.register(
    BotType(
        name="simple_dialin",
        settings_creator=create_simple_dialin_settings,
        required_settings=["dialin_settings"],
        incompatible_with=["call_transfer", "simple_dialout", "voicemail_detection", "silence_detection"],
        auto_add_settings={"dialin_settings": {}},
    )
)

bot_registry.register(
    BotType(
        name="simple_dialout",
        settings_creator=create_simple_dialout_settings,
        required_settings=["dialout_settings"],
        incompatible_with=["call_transfer", "simple_dialin", "voicemail_detection", "silence_detection"],
        auto_add_settings={"dialout_settings": [{}]},
    )
)

bot_registry.register(
    BotType(
        name="voicemail_detection",
        settings_creator=lambda body: body.get(
            "voicemail_detection", {}
        ),  # No creator function in original code
        required_settings=["dialout_settings"],
        incompatible_with=["call_transfer", "simple_dialin", "simple_dialout", "silence_detection"],
        auto_add_settings={"dialout_settings": [{}]},
    )
)

# Register our new silence detection bot
bot_registry.register(
    BotType(
        name="silence_detection",
        settings_creator=create_silence_detection_settings,
        required_settings=["dialin_settings"],
        incompatible_with=["call_transfer", "simple_dialin", "simple_dialout", "voicemail_detection"],
        auto_add_settings={"dialin_settings": {}},
    )
)
