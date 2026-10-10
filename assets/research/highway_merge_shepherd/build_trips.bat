@echo off
setlocal

if "%~1"=="" (
    echo Usage: build_trips.bat ^<light^|medium^|dense^>
    echo   light  = 58 vehicles over 1200 s
    echo   medium = 220 vehicles over 400 s
    echo   dense  = 300 vehicles over 300 s
    exit /b 1
)

set "LEVEL=%~1"
set "BEGIN=0"

if /i "%LEVEL%"=="light" (
    set "COUNT=58"
    set "END=1200"
) else if /i "%LEVEL%"=="medium" (
    set "COUNT=220"
    set "END=400"
) else if /i "%LEVEL%"=="dense" (
    set "COUNT=300"
    set "END=300"
) else (
    echo Unknown level: %LEVEL%
    exit /b 1
)

if not exist "osm.net.xml.gz" (
    echo Missing osm.net.xml.gz. Run netconvert or copy the network file first.
    exit /b 1
)

for /f %%i in ('python -c "print((%END% - %BEGIN% + 1) / %COUNT%)"') do set "PERIOD=%%i"

echo Building %LEVEL% traffic demand (count=%COUNT%, window=%BEGIN%-%END%s, period=%PERIOD%s)...

python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz ^
  --fringe-factor 10 -p %PERIOD% ^
  -o osm.passenger.%LEVEL%.trips.xml -r osm.passenger.%LEVEL%.rou.xml ^
  -b %BEGIN% -e %END% ^
  --trip-attributes "departLane=\"best\"" ^
  --fringe-start-attributes "departSpeed=\"max\"" ^
  --validate --remove-loops ^
  --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link ^
  --vehicle-class passenger --vclass passenger --prefix veh_%LEVEL%_ ^
  --min-distance 300 --min-distance.fringe 10 --allow-fringe.min-length 1000 ^
  --lanes --seed 42

if errorlevel 1 (
    echo Trip generation failed for %LEVEL%.
    exit /b 1
)

echo Done: osm.passenger.%LEVEL%.trips.xml
