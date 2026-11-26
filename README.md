# CSCI-265-Go-Project


*This repository is built and tested with python 3.12*

### 1. Clone and enter the project

```bash
git https://github.com/ntmdrgl/CSCI-265-Go-Project
cd CSCI-265-Go-Project
```

### 2. Create and activate a virtual environment

**Windows (PS):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the activation script, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry. *
> 
> \* https://stackoverflow.com/a/4038991

**Windows (CMD):**
```bat
py -3.12 -m venv .venv
.venv\Scripts\Activate.bat
```

**Linux/MacOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

*Must be done in this order, as some packages may depend on torch. Also note that Torch is pinned to a specific CUDA build - 12.8*

```bash
pip install -r requirements-torch.txt
pip install -r requirements.txt