import yt_dlb
from pydub import AudioSegment
import os

DOWNLOAD_DIR='downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok=True)

def download_youtube_audio(url :str) -> str:
    output_path=os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts={
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "prefferedcodec": "wav",
                "prefferedquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url, download=True)
        filename=ydl.prepare_filename(info).replace(".webm",".wav").replace(".mpa",".wav")
    return filename
print(download_youtube_audio("https://www.youtube.com/watch?v=xlYJhtL0qbQ&t=643s"))