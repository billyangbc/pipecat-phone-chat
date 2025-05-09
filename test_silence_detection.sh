#!/bin/bash

# Start the bot runner in the background
echo "Starting bot runner..."
python bot_runner.py --host localhost &
BOT_RUNNER_PID=$!

# Wait for the bot runner to start
echo "Waiting for bot runner to start..."
sleep 3

# Test the silence detection bot in test mode
echo "Testing silence detection bot in test mode..."
curl -X POST "http://localhost:7860/start" \
     -H "Content-Type: application/json" \
     -d '{
         "config": {
            "silence_detection": {
               "testInPrebuilt": true,
               "silenceThreshold": 10.0,
               "maxUnansweredPrompts": 3
            }
         }
      }'

echo ""
echo "If successful, you should see a response with a room URL."
echo "Open the room URL in a browser to test the silence detection bot."
echo "The bot will prompt you after 10 seconds of silence."
echo "After 3 unanswered prompts, the bot will terminate the call."
echo ""
echo "Press Ctrl+C to stop the bot runner when done testing."

# Wait for user to press Ctrl+C
wait $BOT_RUNNER_PID
