@echo off
SETLOCAL EnableDelayedExpansion
for /F "tokens=*" %%A in (Photos.txt) do (
echo %%~nA %%~xA
for %%B in ("%%A\..") do (
mkdir "D:\Photos\tmp_photos\%%~nxB" 2>nul
C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -normalize -sigmoidal-contrast 2.5 -colorspace HCL -channel g -sigmoidal-contrast 2.5,0%% +channel -colorspace sRGB +repage  -colorspace RGB -filter Lanczos -define filter:lobes=2 -define filter:blur=0.88451002338585141 -resize "2200x2200>" -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA.jpg"
REM C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -normalize -sigmoidal-contrast 3 -colorspace HCL -channel g -sigmoidal-contrast 2,0%% +channel -colorspace sRGB +repage  -colorspace RGB -filter Lanczos -define filter:lobes=2 -define filter:blur=0.88451002338585141 -resize "2200x2200>" -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA_norm4.jpg"
REM C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -normalize -colorspace HCL -channel g -sigmoidal-contrast 3,0%% +channel -colorspace sRGB +repage  -colorspace RGB -filter Lanczos -define filter:lobes=2 -define filter:blur=0.88451002338585141 -resize "2200x2200>" -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA_norm3.jpg"
REM C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -normalize -sigmoidal-contrast 3 -colorspace RGB -filter Lanczos -define filter:lobes=2 -define filter:blur=0.88451002338585141 -resize "2200x2200>" -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA_norm2.jpg"
)
)
rem C:\PortableApps\ImageMagick-7.0.8-10-portable-Q16-x64\convert.exe "%%~A" -colorspace RGB -filter Lanczos -define filter:lobes=4 -define filter:blur=0.88451002338585141 -resize 2200x2200\> -colorspace sRGB -quality 92 "D:\Photos\tmp_photos\%%~nxB\%%~nA.jpg" 
ENDLOCAL
