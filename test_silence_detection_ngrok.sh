#!/bin/bash

# Check if ngrok domain is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <your-ngrok-domain>"
  echo "Example: $0 yourdomain.ngrok.app"
  exit 1
fi

NGROK_DOMAIN=$1

# Start ngrok in the background
echo "Starting ngrok with domain $NGROK_DOMAIN..."
ngrok http --domain $NGROK_DOMAIN 7860 &
NGROK_PID=$!

# Wait for ngrok to start
echo "Waiting for ngrok to start..."
sleep 3

# Start the bot runner
echo "Starting bot runner..."
python bot_runner.py --host localhost &
BOT_RUNNER_PID=$!

# Wait for the bot runner to start
echo "Waiting for bot runner to start..."
sleep 3

echo ""
echo "==================================================="
echo "Ngrok URL: https://$NGROK_DOMAIN"
echo "Webhook URL: https://$NGROK_DOMAIN/start"
echo "==================================================="
echo ""
echo "Configure your Daily phone number to send webhooks to the above URL."
echo "When someone calls your Daily phone number, the silence detection bot will answer."
echo ""
echo "You can also test with a curl command:"
echo ""
echo "curl -X POST \"https://$NGROK_DOMAIN/start\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{
         \"config\": {
            \"silence_detection\": {
               \"testInPrebuilt\": true,
               \"silenceThreshold\": 10.0,
               \"maxUnansweredPrompts\": 3
            },
            \"dialin_settings\": {
               \"from\": \"+12345678901\",
               \"to\": \"+19876543210\",
               \"call_id\": \"test-call-id\",
               \"call_domain\": \"test-call-domain\"
            }
         }
      }'"
echo ""
echo "The bot will prompt the caller after 10 seconds of silence."
echo "After 3 unanswered prompts, the bot will terminate the call."
echo "A call summary will be displayed in the console."
echo ""
echo "Press Ctrl+C to stop the bot runner and ngrok when done testing."

# Wait for user to press Ctrl+C
trap "kill $NGROK_PID $BOT_RUNNER_PID; exit" INT
wait
