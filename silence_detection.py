#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import time
from datetime import datetime

from call_connection_manager import CallConfigManager, CallFlowState, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


class SilenceDetector(FrameProcessor):
    """Detects silence and triggers TTS prompts after a specified duration."""

    def __init__(self, silence_threshold=10.0, max_unanswered_prompts=3):
        """Initialize the silence detector.
        
        Args:
            silence_threshold: Seconds of silence before triggering a prompt
            max_unanswered_prompts: Maximum number of unanswered prompts before terminating
        """
        super().__init__()
        self.silence_threshold = silence_threshold
        self.max_unanswered_prompts = max_unanswered_prompts
        self.last_user_activity = time.time()
        self.user_speaking = False
        self.unanswered_prompts = 0
        self.silence_events = []
        self.call_start_time = time.time()
        self.call_end_time = None
        self.prompt_triggered = False
        self.terminate_call_function = None

    def register_terminate_function(self, terminate_function):
        """Register the terminate call function."""
        self.terminate_call_function = terminate_function

    async def process_frame(self, frame, direction):
        """Process incoming frames to detect silence and user activity."""
        await super().process_frame(frame, direction)

        current_time = time.time()
        
        # Handle user speaking events
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
            self.last_user_activity = current_time
            self.prompt_triggered = False
            logger.debug("User started speaking")
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_speaking = False
            self.last_user_activity = current_time
            logger.debug("User stopped speaking")
            
        elif isinstance(frame, InputAudioRawFrame):
            # Only check for silence if the user is not currently speaking
            if not self.user_speaking:
                silence_duration = current_time - self.last_user_activity
                
                # If silence exceeds threshold and we haven't triggered a prompt yet
                if silence_duration >= self.silence_threshold and not self.prompt_triggered:
                    self.prompt_triggered = True
                    self.unanswered_prompts += 1
                    
                    # Record silence event
                    self.silence_events.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "duration": silence_duration,
                        "prompt_number": self.unanswered_prompts
                    })
                    
                    logger.info(f"Silence detected for {silence_duration:.2f} seconds. Triggering prompt #{self.unanswered_prompts}")
                    
                    # Trigger silence prompt
                    await self.trigger_silence_prompt()
                    
                    # Check if we've reached the maximum number of unanswered prompts
                    if self.unanswered_prompts >= self.max_unanswered_prompts:
                        logger.warning(f"Maximum unanswered prompts ({self.max_unanswered_prompts}) reached. Terminating call.")
                        self.call_end_time = time.time()
                        await self.terminate_call()

        await self.push_frame(frame, direction)

    async def trigger_silence_prompt(self):
        """Trigger a silence prompt."""
        if self.unanswered_prompts == 1:
            prompt = "I noticed you've been quiet for a while. Are you still there?"
        elif self.unanswered_prompts == 2:
            prompt = "I haven't heard from you. If you're still there, please say something."
        else:
            prompt = "Since I haven't heard from you, I'll be ending the call soon. Please speak if you'd like to continue."
        
        # Create a user message frame with the silence prompt
        silence_message = {"role": "user", "content": f"[SILENCE_PROMPT_{self.unanswered_prompts}]"}
        await self.push_frame(silence_message, FrameDirection.UPSTREAM)

    async def terminate_call(self):
        """Terminate the call after maximum unanswered prompts."""
        if self.terminate_call_function:
            await self.terminate_call_function(None)

    def generate_call_summary(self):
        """Generate a summary of the call statistics."""
        if not self.call_end_time:
            self.call_end_time = time.time()
            
        call_duration = self.call_end_time - self.call_start_time
        
        summary = {
            "call_duration_seconds": round(call_duration, 2),
            "call_duration_formatted": f"{int(call_duration // 60)}m {int(call_duration % 60)}s",
            "silence_events": len(self.silence_events),
            "silence_events_details": self.silence_events,
            "unanswered_prompts": self.unanswered_prompts,
            "call_terminated_by_silence": self.unanswered_prompts >= self.max_unanswered_prompts,
            "call_start_time": datetime.fromtimestamp(self.call_start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "call_end_time": datetime.fromtimestamp(self.call_end_time).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Enhanced Dial-in Bot with Silence Detection",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Log call summary before ending
        if silence_detector:
            summary = silence_detector.generate_call_summary()
            logger.info(f"Call Summary: {summary}")
            
            # Print a formatted summary to the console
            print("\n" + "="*50)
            print("CALL SUMMARY")
            print("="*50)
            print(f"Call Duration: {summary['call_duration_formatted']}")
            print(f"Call Start: {summary['call_start_time']}")
            print(f"Call End: {summary['call_end_time']}")
            print(f"Silence Events: {summary['silence_events']}")
            print(f"Unanswered Prompts: {summary['unanswered_prompts']}")
            print(f"Call Terminated by Silence: {summary['call_terminated_by_silence']}")
            
            if summary['silence_events'] > 0:
                print("\nSilence Event Details:")
                for i, event in enumerate(summary['silence_events_details']):
                    print(f"  Event {i+1}: {event['timestamp']} - {event['duration']:.2f}s - Prompt #{event['prompt_number']}")
            print("="*50 + "\n")

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ SILENCE DETECTOR SETUP ------------
    
    # Get silence detection settings if present
    silence_settings = {}
    if "silence_detection" in body:
        silence_settings = body.get("silence_detection", {})
    
    # Get silence threshold and max unanswered prompts from settings or use defaults
    silence_threshold = silence_settings.get("silenceThreshold", 10.0)
    max_unanswered_prompts = silence_settings.get("maxUnansweredPrompts", 3)
    
    logger.info(f"Silence detection configured with threshold: {silence_threshold}s, max unanswered prompts: {max_unanswered_prompts}")
    
    # Initialize the silence detector
    silence_detector = SilenceDetector(
        silence_threshold=silence_threshold,
        max_unanswered_prompts=max_unanswered_prompts
    )
    silence_detector.register_terminate_function(terminate_call)

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.

    When you see a message like [SILENCE_PROMPT_1], [SILENCE_PROMPT_2], or [SILENCE_PROMPT_3], it means the user has been silent for a while. Respond with a friendly prompt asking if they're still there.

    If the user ends the conversation or if you receive a [SILENCE_PROMPT_3] message, **IMMEDIATELY** call the `terminate_call` function.
    """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            silence_detector,   # Silence detection
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        
        # Generate and log call summary
        if silence_detector:
            silence_detector.call_end_time = time.time()
            summary = silence_detector.generate_call_summary()
            logger.info(f"Call Summary (participant left): {summary}")
            
            # Print a formatted summary to the console
            print("\n" + "="*50)
            print("CALL SUMMARY (Participant Left)")
            print("="*50)
            print(f"Call Duration: {summary['call_duration_formatted']}")
            print(f"Call Start: {summary['call_start_time']}")
            print(f"Call End: {summary['call_end_time']}")
            print(f"Silence Events: {summary['silence_events']}")
            print(f"Unanswered Prompts: {summary['unanswered_prompts']}")
            print(f"Call Terminated by Silence: {summary['call_terminated_by_silence']}")
            
            if summary['silence_events'] > 0:
                print("\nSilence Event Details:")
                for i, event in enumerate(summary['silence_events_details']):
                    print(f"  Event {i+1}: {event['timestamp']} - {event['duration']:.2f}s - Prompt #{event['prompt_number']}")
            print("="*50 + "\n")
            
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Dial-in Bot with Silence Detection")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
