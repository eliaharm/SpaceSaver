@echo off
SETLOCAL EnableDelayedExpansion
echo ping.exe  > commands.txt
for /F "tokens=*" %%A in (Photos.txt) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
mkdir "C:\Photos\tmp_photos\%%~nxB" 2>nul
 echo magick convert "%%~A"  -colorspace RGB -filter Lanczos  -define filter:blur=0.88451002338585141  -define filter:lobes=2 -resize "2200x2200>" -colorspace sRGB -normalize -sigmoidal-contrast 1.5 -colorspace HCL -channel g -sigmoidal-contrast 1.0,0%% +channel -colorspace sRGB +repage  -quality 92 "C:\Photos\tmp_photos\%%~nxB\%%~nA.jpg"  >> commands.txt

)
)

C:\PortableApps\mparallel\MParallel.exe --count=8  --input=commands.txt

ENDLOCAL