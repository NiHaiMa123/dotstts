Option Explicit

Dim fso, shell, root, pythonExe, launcher, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = fso.BuildPath(root, ".venv\Scripts\python.exe")
launcher = fso.BuildPath(root, "scripts\launch_voice_webui.py")

If Not fso.FileExists(pythonExe) Then
    MsgBox "Project Python was not found: " & pythonExe, vbCritical, "TTS startup error"
    WScript.Quit 1
End If

If Not fso.FileExists(launcher) Then
    MsgBox "WebUI launcher was not found: " & launcher, vbCritical, "TTS startup error"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
command = Chr(34) & pythonExe & Chr(34) & " " & Chr(34) & launcher & Chr(34)
' Window style 1: visible console IS the server process; closing it stops the server.
shell.Run command, 1, False
