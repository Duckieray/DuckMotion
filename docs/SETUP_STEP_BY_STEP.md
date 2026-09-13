# DuckMotion Setup Guide (For Complete Beginners)

DuckMotion adds **local text/image-to-video generation** to WebbDuck. It is a
plugin, not a standalone app, so this guide assumes WebbDuck is already
installed and running.

> **Do this first:** if you have not installed WebbDuck yet, follow its own
> complete-beginners guide first, get it running, then come back here:
>
> **[WebbDuck Setup Guide (For Complete Beginners)](https://github.com/Duckieray/WebbDuck/blob/main/docs/SETUP.md)**
>
> That guide covers everything this one does **not** re-explain: installing
> Python, downloading WebbDuck, unzipping it, opening a terminal, activating
> its virtual environment, installing PyTorch, and starting the app.

---

## What you will need

- WebbDuck installed and running on **Windows 10/11** or **Linux** (WSL works too).
- **An NVIDIA graphics card (GPU).** WebbDuck can run on CPU for images, but
  DuckMotion video models realistically need a real GPU. **At least 12 GB of
  VRAM is recommended** for good results.
- An internet connection and **30 GB+ of free disk space** (model files and the
  AI runtimes are large).
- **Git installed.** One of DuckMotion's video backends needs the `git` command
  to download its runtime. On Windows install Git from <https://git-scm.com>;
  on Linux `sudo apt install git`.
- A folder with video models (or an intention to download some). This is
  DuckMotion's "model library". It can be a fresh empty folder.

> You do **not** need a GitHub account. You do **not** need any paid software.
> You do **not** need to know anything about `git`, CUDA, or AI frameworks.

---

## Part 1 — Download DuckMotion (no GitHub account needed)

1. Open your browser and go to: **[https://github.com/Duckieray/DuckMotion](https://github.com/Duckieray/DuckMotion)**
2. Click the green **Code** button near the top right of the page.
3. In the menu that drops down, click **Download ZIP**.
4. Wait for `DuckMotion-main.zip` to finish downloading (it goes to your **Downloads** folder).
5. Unzip it, just like you did for WebbDuck:
   - **Windows:** right-click `DuckMotion-main.zip` → **Extract All...** → **Extract**.
   - **Linux:** paste `cd ~/Downloads && unzip DuckMotion-main.zip`.
6. A folder named `DuckMotion-main` appears. Rename it to `duckmotion`.
7. Put the `duckmotion` folder **next to your `webbduck` folder**, so they are
   siblings in the same parent folder. For example `C:\webbduck` and
   `C:\duckmotion` on Windows, or `~/webbduck` and `~/duckmotion` on Linux.
   (Any location works as long as we pass `--webbduck-dir` below; siblings are
   just tidy.)

---

## How to read the commands in this guide

- Click inside the box, press **Ctrl + C** (Windows) or **Ctrl + Shift + C**
  (Linux) to copy it.
- Click in your terminal and paste with **Ctrl + V** / **Ctrl + Shift + V**.
- Press **Enter**.
- Do **not** type the `PS C:\>` or `$` prompt symbols — only paste what is
  inside the boxes.

---

## Part 2 — Windows (PowerShell)

### Step 1: Open PowerShell and use WebbDuck's Python

Press the **Windows key**, type `PowerShell`, and open **Windows PowerShell**.
Then activate the same "virtual environment" WebbDuck uses, so DuckMotion
shares its Python:

```
cd C:\webbduck
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

> If you get a red "scripts are disabled on this system" error, paste this
> first, press **Enter** (press **Y** if asked), then run the `Activate.ps1`
> line again:
>
> ```
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### Step 2: Go to the DuckMotion folder

```
cd C:\duckmotion
```

Check you are in the right place:

```
dir
```

You should see `plugin.json`, a `tools/` folder, a `ui/` folder, and several
`.py` files. If you see that, you are in the right folder.

### Step 3: Point at your model folder and install DuckMotion

You need a folder for video models. Either use one you already have, or create
one now (in File Explorer, make `C:\models`). Then paste this, changing
`C:\models` if yours has another path:

```
python tools\setup.py --models C:\models --webbduck-dir C:\webbduck
```

Then press **Enter**. This one command does everything:

1. Saves your model folder as DuckMotion's shared model library.
2. Downloads and builds DuckMotion's **isolated video runtimes** (Wan, LTX,
   and LTX ConvRot). This is the big download — several gigabytes of AI
   libraries — so it can take a while. **Do not close the window.**
3. Prepares support files and recipes for any video models it found.
4. Installs the DuckMotion plugin into WebbDuck
   (`C:\webbduck\plugins\webapps\duckmotion`).

When it finishes you should see **`Setup complete.`** and a `Next:` line.

### Step 4: Run the DuckMotion "doctor" check

```
python tools\doctor.py
```

This never loads models or generates video — it only checks that everything is
in place and prints a simple report:

- Under **Runtimes**, each of `wan`, `ltx25`, and `ltx25_convrot` should show a ✓.
- Under **Models discovered**, video models in your model folder (or Hugging
  Face cache) are listed with ✓ or ✗ per-model.

A ✓ next to a model means it is ready to generate. If a model shows ✗, the
reason is printed under it (for example a missing recipe or a missing support
asset) — rerun `setup.py` after fixing it.

### Step 5: Restart WebbDuck and use it

In the window where WebbDuck is running, press **Ctrl + C**, then start it again:

```
cd C:\webbduck
.\.venv\Scripts\Activate.ps1
python .\run.py --output .\outputs\ --port 8010
```

Then open **[http://localhost:8010](http://localhost:8010)**. Look for the
**DuckMotion** tab and select it. Pick a video model, write a prompt, and
generate.

> Keep these terminal windows open while you use the app. Video generation is
> slow (minutes per clip), and it locks the GPU while running — that is normal.

---

## Part 3 — Linux (and WSL)

### Step 1: Open a terminal and use WebbDuck's Python

Press **Ctrl + Alt + T** (or open your WSL shell), then activate WebbDuck's
virtual environment:

```
cd ~/webbduck
source .venv/bin/activate
```

Your prompt should start with `(.venv)`.

### Step 2: Go to the DuckMotion folder

```
cd ~/duckmotion
```

Check you are in the right place:

```
ls
```

You should see `plugin.json`, a `tools/` folder, a `ui/` folder, and several
`.py` files.

### Step 3: Point at your model folder and install DuckMotion

Make a model folder if you do not have one (for example `mkdir ~/models`), then
paste this, changing `~/models` if yours has another path:

```
python tools/setup.py --models ~/models --webbduck-dir ~/webbduck
```

Press **Enter**. This one command does everything:

1. Saves your model folder as DuckMotion's shared model library.
2. Downloads and builds DuckMotion's **isolated video runtimes** (Wan, LTX,
   and LTX ConvRot). Several gigabytes of AI libraries — **do not close the
   terminal.**
3. Prepares support files and recipes for any video models it found.
4. Installs the DuckMotion plugin into WebbDuck
   (`~/webbduck/plugins/webapps/duckmotion`).

When it finishes you should see **`Setup complete.`** and a `Next:` line.

### Step 4: Run the DuckMotion "doctor" check

```
python tools/doctor.py
```

This never loads models or generates video — it only checks that everything is
in place and prints a simple report (see Windows Step 4 above for how to read
it).

### Step 5: Restart WebbDuck and use it

In the terminal where WebbDuck is running, press **Ctrl + C**, then start it again:

```
cd ~/webbduck
source .venv/bin/activate
python run.py --output ./outputs --port 8010
```

Then open **[http://localhost:8010](http://localhost:8010)**, open the
**DuckMotion** tab, pick a video model, and generate.

---

## Part 4 — Getting video models

DuckMotion selects the right runtime automatically from the model you pick —
you never choose an "engine". Into your model folder you can put:

- **Wan 2.2** — Diffusers `model_index.json` folders, or `.gguf` weights
  (paired `H.gguf` / `L.gguf` files appear as one model automatically);
- **LTX-2.5** — Diffusers model folders, or INT8 "ConvRot" `.safetensors`
  files that ship with a companion recipe;
- anything already in your ordinary **Hugging Face cache** (DuckMotion scans it too).

Restart WebbDuck after adding files, then reload the DuckMotion tab. Run
`python tools/doctor.py` any time to see exactly why a model is (not) ready.

For the full detail on formats, provenance, recipes, and execution profiles,
read `docs/LTX25_CONVROT.md` and `docs/EXECUTION_RECIPES.md`.

---

## Restarting and updating DuckMotion

**Restart later:** just restart WebbDuck (see Windows/Linux Step 5 above).
DuckMotion loads automatically.

**Update to a newer DuckMotion:**

1. Download the latest ZIP and unzip it into a new `duckmotion` folder
   (keep the old one for comparison if you like).
2. Run the one command again from the new folder (Windows/Linux Step 3), then
   `tools\doctor.py` / `tools/doctor.py`.
3. Restart WebbDuck. Models and outputs stay where they are — DuckMotion only
   reads them, so nothing moves or is lost.

---

## Quick guide: common problems

| If you see this... | It means... | Do this... |
| --- | --- | --- |
| `python is not recognized` / `Python was not found` | Python is not on your PATH, or you skipped the venv activation | Activate WebbDuck's venv first (Windows/Linux Step 1), and if that fails reinstall Python with **"Add python.exe to PATH"** ticked (see WebbDuck's SETUP guide) |
| `running scripts is disabled on this system` | Windows blocks activation scripts | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again (Windows Step 1) |
| `Model directory does not exist: ...` | The `--models` folder does not exist | Create the folder first, or correct the path, and rerun |
| `git: command not found` / `git is not recognized` | Git is not installed | Install Git (<https://git-scm.com> on Windows, `sudo apt install git` on Linux), reopen the terminal, rerun `setup.py` |
| `CUDA` / `no kernel image is available for execution on the device` | PyTorch build does not match your GPU or driver | Install/update the NVIDIA driver and reboot, then rerun `setup.py` |
| `0 models` / nothing under "Models discovered" | No video models are in your model folder yet | Add Wan/LTX models to the folder (Part 4), rerun `setup.py`, restart WebbDuck |
| A model shows ✗ in doctor | A recipe or support asset is missing | Read the reason printed under the model; it usually says to rerun `setup.py` after fixing the reported issue |
| Setup/doctor prints a long stack trace | Something in the isolated runtimes failed | Copy the **exact red text** — that is the most useful thing to search for or to share |
| Generation is very slow or runs out of memory | Not enough VRAM for the chosen resolution/frames | Pick a smaller model or lower resolution/frames; a 16 GB+ GPU is best for full LTX quality |

If a command fails, the exact red text it prints is the most useful thing to
search for or share with someone helping you.

---

## More help

- `README.md` — product overview, architecture rules, and the full doc list.
- `docs/MODEL_DRIVEN_VIDEO_ARCHITECTURE.md` — how models drive the runtime.
- `docs/LTX25_CONVROT.md` — the ConvRot execution profile, assets, and defaults.
- `docs/EXECUTION_RECIPES.md` — recipe/execution-profile contracts.
- `docs/HARDWARE_SMOKE_MATRIX.md` — hardware validation matrix.