echo ":: Switching to M:/home/repos/reddit-bot-gpt2-xl"
# Change directory
cd /mnt/m/home/repos/reddit-bot-gpt2-xl

echo ":: Starting Bot!"
# Loop indefinitely
while true; do
    python run-bot-text.py

    # Capture the exit code of the last command
    exit_code=$?

    if [ $exit_code -eq 1 ]; then
        echo "Python script exited with code 1, restarting..."
        continue
    else
        echo "Python script completed successfully."
        break
    fi
done