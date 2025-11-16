v@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 입력 폴더 (한글 경로 가능)
set INPUT_DIR=C:\Users\11e26\Desktop\internship\wake-word-benchmark-master\audio\alexa

REM 출력 폴더
set OUTPUT_DIR=%INPUT_DIR%\converted
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM 변환용 카운터
set /a COUNT=1

REM FLAC → WAV 변환, 순서대로 파일명 변경
for %%f in ("%INPUT_DIR%\*.flac") do (
    set "NEWNAME=alexa_sample!COUNT!.wav"
    echo 변환 중: %%f → !NEWNAME!
    ffmpeg -y -i "%%f" -ar 8000 -ac 1 -acodec pcm_s16le "%OUTPUT_DIR%\!NEWNAME!"
    set /a COUNT+=1
)

echo 변환 완료!
pause
