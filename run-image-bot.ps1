Set-Location M:\

Write-Output ":: Starting gpt-diffusion-bot-3-11 env"

.\envs\gpt-diffusion-bot-3-11\Scripts\activate

Write-Output ":: Switching to D:\code\repos\reddit-bot-gpt2-xl"

Set-Location D:\code\repos\reddit-bot-gpt2-xl

Write-Output ":: Starting Image Bot!"
while ($true) {
    python run-bot-image.py

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