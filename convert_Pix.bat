@echo off
SETLOCAL EnableDelayedExpansion
for /F "tokens=*" %%A in (Photos.txt) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
mkdir "D:\Photos\tmp_photos\%%~nxB" 2>nul
C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -colorspace RGB -filter Lanczos -define filter:lobes=4 -define filter:blur=0.88451002338585141 -resize "2200x2200>" -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA.jpg"
)
)
rem C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -colorspace RGB -filter Lanczos -define filter:lobes=4 -define filter:blur=0.88451002338585141 -resize 2200x2200\> -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA.jpg" 
ENDLOCAL