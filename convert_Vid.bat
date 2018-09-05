@echo off
for /F "tokens=*" %%A in (videos.txt) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
REM mkdir "D:\Photos\tmp_videos\%%~nxB" 2>nul
C:\PortableApps\ffmpeg-4.0.2-win64-shared\bin\ffmpeg.exe  -i "%%~A" -s hd720 -c:v libx265 -pix_fmt yuv420p -crf 23 -c:a aac "D:\Photos\tmp_videos\%%~nxB_%%~nA_cpu2.mkv"
REM C:\PortableApps\ffmpeg-4.0.2-win64-shared\bin\ffmpeg.exe -i "%%~A" -s hd720 -c:v hevc_nvenc -pix_fmt yuv420p -preset slow -c:a aac "D:\Photos\tmp_videos\%%~nxB-%%~nA.mkv"
REM -pix_fmt yuv420p  (or) -y -vf "scale=in_range=pc:out_range=tv"  for color range conversion from full to limited
REM -pix_fmt p010le for YUV 4:2:0 10-bit
)
)
