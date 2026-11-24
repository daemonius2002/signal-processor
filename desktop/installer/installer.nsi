Unicode true
!include "MUI2.nsh"

!define APP_NAME "Signal Processor"
!define APP_VERSION "1.4.2"
!define APP_DIR "SignalProcessor"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\SignalProcessor"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "SignalProcessor_Setup_v${APP_VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\${APP_DIR}"
InstallDirRegKey HKCU "${UNINST_KEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 64
ShowInstDetails nevershow

!define MUI_ICON "app_root\app\ui\icon.ico"
!define MUI_UNICON "app_root\app\ui\icon.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_RUN_TEXT "Pornește Signal Processor"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Romanian"

Function LaunchApp
  SetOutPath "$INSTDIR\app"
  Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\app\main.py"'
FunctionEnd

; ---------------------------------------------------------------------------
; Running-application check (v1.4.2)
; Windows refuses to overwrite python313.dll / pythonw.exe while the program
; runs. The check opens the runtime DLL for writing: that fails exactly when a
; copy of the program started from $INSTDIR is alive.
; ---------------------------------------------------------------------------
!macro RUNNING_CHECK UN
Function ${UN}IsAppRunning
  ; result on the stack: "1" running, "0" not running
  IfFileExists "$INSTDIR\python\python313.dll" 0 notrunning
  ClearErrors
  FileOpen $0 "$INSTDIR\python\python313.dll" a
  IfErrors running
  FileClose $0
  notrunning:
    Push "0"
    Return
  running:
    Push "1"
FunctionEnd

Function ${UN}CloseRunningApp
  Call ${UN}IsAppRunning
  Pop $1
  StrCmp $1 "0" done
  MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "Signal Processor este pornit.$\r$\n$\r$\nApasă OK ca să îl închid și să continui, sau Anulare ca să oprești." /SD IDOK IDOK closeit
  Abort "Oprit: Signal Processor este încă pornit."
  closeit:
    InitPluginsDir
    File "/oname=$PLUGINSDIR\close_app.ps1" "close_app.ps1"
    nsExec::Exec 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PLUGINSDIR\close_app.ps1" -Dir "$INSTDIR"'
    Pop $2
    StrCpy $3 0
  waitloop:
    Call ${UN}IsAppRunning
    Pop $1
    StrCmp $1 "0" done
    IntOp $3 $3 + 1
    IntCmp $3 20 manual 0 manual
    Sleep 250
    Goto waitloop
  manual:
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "Nu am putut închide Signal Processor.$\r$\n$\r$\nÎnchide fereastra programului (sau oprește pythonw.exe din Task Manager), apoi apasă Reîncercare." /SD IDCANCEL IDRETRY retry
    Abort "Oprit: Signal Processor este încă pornit."
  retry:
    StrCpy $3 0
    Goto waitloop
  done:
FunctionEnd
!macroend
!insertmacro RUNNING_CHECK ""
!insertmacro RUNNING_CHECK "un."

Section "Install"
  Call CloseRunningApp
  SetOutPath "$INSTDIR"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\samples_stele"
  RMDir /r "$INSTDIR\samples_sunete"
  File /r "app_root\*"

  SetOutPath "$INSTDIR\app"
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' "$INSTDIR\app\ui\icon.ico"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME} (diagnostic).lnk" "$INSTDIR\python\python.exe" '"$INSTDIR\app\main.py"' "$INSTDIR\app\ui\icon.ico"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Sunete de stele (WAV).lnk" "$INSTDIR\samples_stele"
  CreateShortCut "$SMPROGRAMS\${APP_NAME}\Dezinstalare.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' "$INSTDIR\app\ui\icon.ico"

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\app\ui\icon.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Call un.CloseRunningApp
  Delete "$DESKTOP\${APP_NAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APP_NAME}"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\samples_stele"
  RMDir /r "$INSTDIR\samples_sunete"
  RMDir /r "$INSTDIR\docs"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
SectionEnd
