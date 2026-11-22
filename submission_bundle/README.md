# Samsung XR Hackathon 2026 - Track 03 (Path Finder)

## Overview
IMU-only 6-DoF dead-reckoning estimator (timestamp tx ty tz qx qy qz qw in TUM format) for degraded IMU streams (noise, bias drift, frame dropouts, and 5-second post-failure dead reckoning) under single-thread, <= 8 MB peak RAM, and <= 20 ms/s compute budgets.

## Build (Native C99 - Windows / MSYS2 UCRT64)
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\ucrt64\bin;C:\msys64\ucrt64\bin;C:\msys64\ucrt64\bin;C:\msys64\ucrt64\bin;C:\Program Files\Common Files\Oracle\Java\javapath;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA Corporation\NVIDIA App\NvDLISR;C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;C:\Program Files\nodejs\;C:\Program Files\WSL\;C:\Users\senth\AppData\Local\Programs\Python\Python313\Scripts\;C:\Users\senth\AppData\Local\Programs\Python\Python313\;C:\Users\senth\.dotnet;C:\Users\senth\AppData\Local\Programs\Python\Python311\Scripts\;C:\Users\senth\AppData\Local\Programs\Python\Python311\;C:\Users\senth\AppData\Local\Microsoft\WindowsApps;C:\Users\senth\AppData\Local\Programs\Microsoft VS Code\bin;C:\Users\senth\AppData\Local\Programs\Antigravity IDE\bin;C:\Users\senth\.dotnet\tools;C:\Users\senth\AppData\Roaming\npm;C:\Users\senth\AppData\Local\Programs\Python\Python311\python.exe;C:\Users\senth\AppData\Local\Programs\Amazon\AWSCLIV2\;C:\Users\senth\AppData\Local\Programs\DockerDesktop\resources\bin"
gcc -std=c99 -O2 -Wall -I c_src\include c_src\src\main.c c_src\src\dr_detector.c c_src\src\dr_io.c c_src\src\dr_math.c -o bin\imu_dr.exe -lm -lpsapi

## Run Commands
- Native C99 (default): .\bin\imu_dr.exe --input <imu0_data.csv> --output estimated_trajectory.txt --metadata metadata.json
- Native C99 (opt-in multi-interval ZUPT): .\bin\imu_dr.exe --input <imu0_data.csv> --output estimated_trajectory.txt --metadata metadata.json --zupt
- Standalone Python: .\.venv\Scripts\python.exe src\main.py --input <imu0_data.csv> --output estimated_trajectory.txt

## Output Format
- File: estimated_trajectory.txt (space-separated, 8 columns: 	imestamp tx ty tz qx qy qz qw)
- Timestamps in Unix seconds (s = ns * 1e-9), one pose per IMU frame.
