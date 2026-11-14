@echo off
setlocal

cd /d "%~dp0"

set "LEVEL=%~1"

if "%LEVEL%"=="" goto menu

if /i "%LEVEL%"=="light" goto run_light
if /i "%LEVEL%"=="medium" goto run_medium
if /i "%LEVEL%"=="dense" goto run_dense

echo Unknown density level: %LEVEL%
echo Valid options: light, medium, dense
exit /b 1

:menu
echo.
echo Traffic density preset:
echo   1 = light  (sparse traffic, density 5)
echo   2 = medium (moderate traffic, density 15) [default]
echo   3 = dense  (heavy traffic, density 30)
echo.
set /p CHOICE="Select density [1-3, default 2]: "

if "%CHOICE%"=="" set "CHOICE=2"
if "%CHOICE%"=="1" goto run_light
if "%CHOICE%"=="2" goto run_medium
if "%CHOICE%"=="3" goto run_dense

echo Invalid choice: %CHOICE%
exit /b 1

:run_light
if not exist "osm.passenger.light.trips.xml" (
    echo Missing osm.passenger.light.trips.xml. Run build_light.bat or build_all.bat first.
    exit /b 1
)
sumo-gui -c osm.light.sumocfg
exit /b 0

:run_medium
if not exist "osm.passenger.medium.trips.xml" (
    echo Missing osm.passenger.medium.trips.xml. Run build_medium.bat or build_all.bat first.
    exit /b 1
)
sumo-gui -c osm.medium.sumocfg
exit /b 0

:run_dense
if not exist "osm.passenger.dense.trips.xml" (
    echo Missing osm.passenger.dense.trips.xml. Run build_dense.bat or build_all.bat first.
    exit /b 1
)
sumo-gui -c osm.dense.sumocfg
exit /b 0
