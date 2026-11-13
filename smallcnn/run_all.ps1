# Train and analyse all small-CNN runs: joint / Posner-only x fixed / fresh noise x seeds.
#   pwsh smallcnn/run_all.ps1                 # seeds 0-2
#   pwsh smallcnn/run_all.ps1 -Seeds 3,4,5    # more seeds later
#   pwsh smallcnn/run_all.ps1 -Tasks joint -Noise fresh -Tag _v2 -TrainArgs "--lr","3e-4","--noise","search=0.12" -Seeds (0..9)
param([int[]]$Seeds = @(0, 1, 2), [string[]]$Tasks = @("joint", "posner"), [string[]]$Noise = @("fixed", "fresh"),
      [string]$Tag = "", [string[]]$TrainArgs = @(), [switch]$SkipLayers)
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
# Block idle sleep while this script runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); closing the lid still sleeps.
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")
foreach ($seed in $Seeds) {
    foreach ($tasks in $Tasks) {
        foreach ($noise in $Noise) {
            $run = "runs/smallcnn/${tasks}_${noise}${Tag}_seed$seed"
            $train = @("--seed", $seed, "--out", $run) + $TrainArgs
            if ($noise -eq "fresh") { $train += "--fresh-noise" }
            if ($tasks -eq "posner") { $train += @("--tasks", "posner") }
            if (-not (Test-Path "$run/model.pt")) { & $py smallcnn/train.py @train; if ($LASTEXITCODE) { throw "train $run" } }
            & $py smallcnn/recalibrate.py --run $run; if ($LASTEXITCODE) { throw "recalibrate $run" }
            if (-not (Test-Path "$run/lesion_within/lesions.csv")) { & $py smallcnn/lesion.py --run $run --decision within; if ($LASTEXITCODE) { throw "lesion $run" } }
            if (-not $SkipLayers -and -not (Test-Path "$run/layers/lesions.csv")) { & $py smallcnn/layer_lesions.py --run $run; if ($LASTEXITCODE) { throw "layers $run" } }
            if (-not (Test-Path "$run/neuron_cueing/units.csv")) { & $py smallcnn/neuron_cueing.py --run $run; if ($LASTEXITCODE) { throw "neuron_cueing $run" } }
        }
    }
}
