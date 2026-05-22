set MOD_PATH=E:\JellyProject\sts2_mod_agent\mods\loudspeakermod

set GODOT_PATH=E:\sts2_recomplie\Godot_v4.5.1-stable_mono_win64\Godot_v4.5.1-stable_mono_win64.exe

set GODOT_EXPORT_PATH=E:\custom.pck

set STS2_PATH=D:\SteamLibrary\steamapps\common\Slay the Spire 2\mods\Sts2Custom\STS2Custom.pck

"%GODOT_PATH%" --headless --path "%MOD_PATH%" --export-pack "Windows Desktop" "%GODOT_EXPORT_PATH%"

cd /d "%MOD_PATH%"

dotnet build

copy "%MOD_PATH%\.godot\mono\temp\bin\Debug\STS2Custom.dll" "D:\SteamLibrary\steamapps\common\Slay the Spire 2\mods\Sts2Custom\"

copy "%GODOT_EXPORT_PATH%" "%STS2_PATH%"

timeout /t 3
