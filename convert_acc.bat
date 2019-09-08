@echo off
H:
cd H:\Music\
for /r %%A in (*.mp3) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
mkdir "H:\Music_acc\%%~nxB" 2>nul
del "artwork.jpg" >nul 2>&1
C:\PortableApps\ffmpeg-4.1.1-win64-shared\bin\ffmpeg-hi10-heaac.exe -i "%%~A" -an -vcodec copy artwork.jpg
C:\PortableApps\ffmpeg-4.1.1-win64-shared\bin\ffmpeg-hi10-heaac.exe  -i "%%~A"  -c:a libfdk_aac -vn "H:\Music_acc\%%~nxB\%%~nA.m4a"
C:\PortableApps\ffmpeg-4.1.1-win64-shared\bin\AtomicParsley-win32-0.9.0\AtomicParsley.exe "H:\Music_acc\%%~nxB\%%~nA.m4a" --artwork artwork.jpg --overWrite
)
)
