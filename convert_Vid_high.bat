@echo off
setlocal enabledelayedexpansion

for /F "tokens=*" %%A in (videos.txt) do (
    echo Processing: %%~nA%%~xA
    for %%B in ("%%A\..") do (
        mkdir "E:\Photos\tmp_videos\%%~nxB"
        ptime ffmpeg -i "%%~A" ^
            -map 0:v:0 ^
            -map 0:a? ^
            -map 0:s? ^
            -vf "scale=-2:1080:flags=lanczos" ^
            -c:v hevc_nvenc ^
            -rc vbr ^
            -cq 25 ^
            -b:v 0 ^
            -preset p5 ^
            -tier high ^
            -pix_fmt yuv420p ^
            -c:a copy ^
            -c:s copy ^
            "E:\Photos\tmp_videos\%%~nxB\%%~nA.mkv"
    )
)