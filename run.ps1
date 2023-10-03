cd D:\code\repos\reddit-bot-gpt2-xl
.\venv\Scripts\activate

while ($true) {
    # Run the Python script
    python run-bot.py --mode=text

    # Check the exit code of the Python script
    if ($LASTEXITCODE -eq 1) {
        Write-Host "Python script exited with code 1, restarting..."
        continue
    }
    else {
        Write-Host "Python script completed successfully."
        break
    }
}
