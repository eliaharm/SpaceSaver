@echo off
for /F "tokens=*" %%A in (videos.txt) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
REM mkdir "E:\Photos\tmp_videos\%%~nxB" 2>nul
ptime ffmpeg  -i "%%~A" -lavfi "[0:v]scale=ih*16/9:-1,boxblur=luma_radius=min(h\,w)/20:luma_power=1:chroma_radius=min(cw\,ch)/20:chroma_power=1[bg];[bg][0:v]overlay=(W-w)/2:(H-h)/2,crop=h=iw*9/16" -s hd720 -c:v libx265 -pix_fmt yuv420p -crf 23 -c:a aac "E:\Photos\tmp_videos\%%~nxB_%%~nA.mkv"
REM ffmpeg -i "%%~A" -s hd720 -c:v hevc_nvenc -pix_fmt yuv420p -preset slow -c:a aac "E:\Photos\tmp_videos\%%~nxB-%%~nA.mkv"
REM -pix_fmt yuv420p  (or) -y -vf "scale=in_range=pc:out_range=tv"  for color range conversion from full to limited
REM -pix_fmt p010le for YUV 4:2:0 10-bit
)
)
