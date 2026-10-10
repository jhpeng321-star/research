@echo off
setlocal

call "%~dp0build_trips.bat" light
if errorlevel 1 exit /b 1

call "%~dp0build_trips.bat" medium
if errorlevel 1 exit /b 1

call "%~dp0build_trips.bat" dense
if errorlevel 1 exit /b 1

echo.
echo All density presets generated:
echo   light  - osm.passenger.light.trips.xml   (58 vehicles over 1200 s)
echo   medium - osm.passenger.medium.trips.xml  (220 vehicles over 400 s)
echo   dense  - osm.passenger.dense.trips.xml   (300 vehicles over 300 s)
