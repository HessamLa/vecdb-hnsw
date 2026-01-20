# VecDB Development Environment Setup

Step-by-step guide for setting up the development environment on Windows 11 with WSL2.

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Windows 11 | 22H2+ | Host OS |
| WSL2 | Latest | Linux environment |
| Docker Desktop | 4.x+ | Container runtime |
| VSCode | Latest | IDE |
| Dev Containers Extension | Latest | VSCode ↔ Docker integration |

---

## Step 1: Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

If WSL is already installed, ensure you're using WSL2:

```powershell
wsl --set-default-version 2
```

Verify installation:

```powershell
wsl --version
```

Expected output should show WSL version 2.x.x.

---

## Step 2: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/

2. Run the installer

3. During installation, ensure **"Use WSL 2 instead of Hyper-V"** is selected

4. After installation, open Docker Desktop

5. Go to **Settings → Resources → WSL Integration**

6. Enable integration with your WSL distro (e.g., Ubuntu)

7. Click **Apply & Restart**

Verify Docker works in WSL:

```bash
# Open WSL terminal (e.g., Ubuntu)
docker --version
docker run hello-world
```

---

## Step 3: Install VSCode and Extensions

1. Download VSCode from: https://code.visualstudio.com/

2. Install the **Dev Containers** extension:
   - Open VSCode
   - Press `Ctrl+Shift+X` (Extensions)
   - Search for "Dev Containers"
   - Install the extension by Microsoft

3. (Optional) Install other recommended extensions:
   - Remote - WSL
   - Docker

---

## Step 4: Clone the Project

Open your WSL terminal:

```bash
# Create projects directory (if needed)
mkdir -p ~/projects
cd ~/projects

# Clone the repository (replace with actual URL)
git clone <repository-url> vecdb
cd vecdb
```

Or if starting fresh:

```bash
mkdir -p ~/projects/vecdb
cd ~/projects/vecdb
# Copy project files here
```

---

## Step 5: Open in Dev Container

### Option A: From VSCode (Recommended)

1. Open VSCode

2. Press `Ctrl+Shift+P` → Type "Dev Containers: Open Folder in Container"

3. Navigate to `\\wsl$\Ubuntu\home\<username>\projects\vecdb`

4. Click **Open**

5. Wait for the container to build (first time takes 2-5 minutes)

6. Once complete, you're inside the container!

### Option B: From WSL Terminal

```bash
cd ~/projects/vecdb
code .
```

Then in VSCode:
- Click the green button in the bottom-left corner
- Select "Reopen in Container"

---

## Step 6: Verify Environment

Once inside the container, open a terminal (`Ctrl+``) and run:

```bash
bash scripts/verify-environment.sh
```

Expected output:

```
========================================
VecDB Environment Verification
========================================

[✓] Python: Python 3.11.x
[✓] NumPy: 1.2x.x
[✓] pybind11: 2.1x.x
[✓] pytest: 7.x.x
[✓] g++: g++ (Ubuntu ...) 11.x.x
[✓] CMake: cmake version 3.2x.x
[✓] pybind11 CMake dir: /usr/local/lib/python3.11/...

========================================
All checks passed!
========================================
```

---

## Directory Structure

After setup, you'll have:

```
Host (WSL): ~/projects/vecdb/     ←→     Container: /workspaces/vecdb/
                                         (same files, bind-mounted)
```

Any changes made in the container are immediately visible on the host, and vice versa.

---

## Common Commands

### Inside the Container

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/python/test_collection.py

# Run tests with coverage
pytest tests/ --cov=src/python/vecdb

# Build C++ module (after CMakeLists.txt is created)
cd build
cmake ..
make -j$(nproc)

# Verify environment
bash scripts/verify-environment.sh
```

### Container Management (from host)

```bash
# Rebuild container (after Dockerfile changes)
# In VSCode: Ctrl+Shift+P → "Dev Containers: Rebuild Container"

# Or from terminal:
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## Troubleshooting

### Container fails to start

```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs vecdb-dev

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
```

### Permission issues with files

```bash
# Inside container, fix ownership
sudo chown -R vscode:vscode /workspaces/vecdb
```

### Python imports not working

```bash
# Verify PYTHONPATH
echo $PYTHONPATH
# Should show: /workspaces/vecdb/src/python

# Reinstall in dev mode
pip install -e .
```

### CMake can't find pybind11

```bash
# Get pybind11 CMake directory
python -m pybind11 --cmakedir

# Use in CMakeLists.txt:
# set(pybind11_DIR "/path/from/above")
```

### Docker Desktop not starting on Windows

1. Ensure virtualization is enabled in BIOS
2. Run in PowerShell (Admin): `bcdedit /set hypervisorlaunchtype auto`
3. Restart Windows

---

## Development Workflow

### Daily Workflow

1. Open VSCode
2. Click green button → "Reopen in Container" (or it auto-reconnects)
3. Make changes to code
4. Run tests: `pytest tests/`
5. Changes auto-save to host filesystem

### After Pulling New Changes

```bash
# If Dockerfile changed, rebuild:
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"

# If requirements changed:
pip install -r requirements-dev.txt
```

### Stopping Work

Just close VSCode. The container stops automatically.

To manually stop:

```bash
# From host terminal
docker-compose down
```

---

## File Locations Reference

| What | Host (WSL) | Container |
|------|------------|-----------|
| Project root | `~/projects/vecdb/` | `/workspaces/vecdb/` |
| Python source | `~/projects/vecdb/src/python/` | `/workspaces/vecdb/src/python/` |
| C++ source | `~/projects/vecdb/src/cpp/` | `/workspaces/vecdb/src/cpp/` |
| Tests | `~/projects/vecdb/tests/` | `/workspaces/vecdb/tests/` |
| Build output | `~/projects/vecdb/build/` | `/workspaces/vecdb/build/` |

---

## Next Steps

After environment setup is complete:

1. Review `docs/PRD.md` for project requirements
2. Review `docs/ORCHESTRATION.md` for task assignments
3. Begin development following the orchestration workflow

---

*End of Setup Guide*
