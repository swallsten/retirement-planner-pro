# Retirement Planner Pro

A Monte Carlo retirement simulation dashboard that helps you stress-test whether your savings will last through retirement. It runs 10,000 randomized market simulations against your actual portfolio and spending plan, then shows you the odds.

## What It Does

- **Monte Carlo simulation** — runs thousands of market scenarios (stocks, bonds, REITs, inflation) to estimate the probability your money lasts
- **Portfolio upload** — import CSV exports from Schwab, Fidelity, Vanguard, etc. and it auto-classifies your holdings
- **Sensitivity analysis** — tornado chart shows which assumptions matter most (spending, retirement age, returns, etc.)
- **Spending phases** — model different spending levels for early/mid/late retirement
- **Stress tests** — simulate market crashes at retirement, sequence-of-returns risk
- **Tax-aware** — models Roth conversions, RMDs, IRMAA surcharges, capital gains, tax-loss harvesting
- **Social Security optimization** — test different claiming ages for you and a spouse
- **Adaptive spending** — optional Guyton-Klinger guardrails that adjust withdrawals based on portfolio performance

## Quick Start (If You Already Have Python)

```bash
pip install -r requirements.txt
streamlit run retirement_mc_pro.py
```

A browser window opens automatically. That's it.

---

## Installing Python From Scratch

If you've never used Python before, follow the instructions for your operating system below. This takes about 10 minutes.

### Mac

**Step 1: Open Terminal**

Press **Cmd + Space**, type **Terminal**, and hit Enter. A black/white window with a command prompt appears. All the commands below get typed here.

**Step 2: Install Homebrew (a Mac package manager)**

Copy and paste this entire line into Terminal, then press Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your Mac login password (you won't see characters as you type — that's normal). It may take a few minutes. If it says Homebrew is already installed, that's fine — move on.

After it finishes, it may print instructions saying to run two commands starting with `echo` and `eval` to add Homebrew to your PATH. If so, copy and paste those lines into Terminal and press Enter after each one.

**Step 3: Install Python**

```bash
brew install python
```

Verify it worked:

```bash
python3 --version
```

You should see something like `Python 3.12.x` or `Python 3.13.x`. Any version 3.9 or higher is fine.

**Step 4: Install the app's dependencies**

Navigate to the folder where the app files are. For example, if they're in your Downloads folder:

```bash
cd ~/Downloads
```

Or if they're somewhere else, replace the path:

```bash
cd /path/to/the/folder/containing/retirement_mc_pro.py
```

Then install the required packages:

```bash
pip3 install -r requirements.txt
```

**Step 5: Run the app**

```bash
streamlit run retirement_mc_pro.py
```

Your browser will open automatically to `http://localhost:8501` with the dashboard. To stop the app later, go back to Terminal and press **Ctrl + C**.

---

### Windows

**Step 1: Download Python**

Go to [python.org/downloads](https://www.python.org/downloads/) and click the big yellow **"Download Python 3.x.x"** button.

**Step 2: Run the installer**

Open the downloaded file. On the very first screen of the installer:

> **IMPORTANT:** Check the box at the bottom that says **"Add python.exe to PATH"** before clicking anything else.

Then click **"Install Now"**. Wait for it to finish, then click "Close."

**Step 3: Open Command Prompt**

Press **Windows key**, type **cmd**, and hit Enter. A black window appears.

Verify Python installed correctly:

```cmd
python --version
```

You should see `Python 3.x.x`. If you get an error saying "python is not recognized," you probably missed the "Add to PATH" checkbox — rerun the installer, choose "Modify," and make sure PATH is checked.

**Step 4: Navigate to the app folder**

If the files are in your Downloads folder:

```cmd
cd %USERPROFILE%\Downloads
```

Or wherever the files are:

```cmd
cd C:\path\to\the\folder
```

**Step 5: Install dependencies**

```cmd
pip install -r requirements.txt
```

**Step 6: Run the app**

```cmd
streamlit run retirement_mc_pro.py
```

Your browser opens automatically. To stop the app, go back to Command Prompt and press **Ctrl + C**.

---

## Files You Need

Make sure these files are all in the same folder:

| File | What it is |
|------|-----------|
| `retirement_mc_pro.py` | The main application |
| `requirements.txt` | List of Python packages needed |
| `.streamlit/config.toml` | Theme configuration (light mode, colors) |

The `.streamlit` folder may be hidden by default. On Mac, press **Cmd + Shift + .** in Finder to show hidden files. On Windows, check "Hidden items" in File Explorer's View menu.

## Using the App

### First Time

1. The app opens to the **Results** page showing a simulation with placeholder portfolio values
2. Upload your brokerage CSV exports (taxable and/or retirement accounts) using the upload boxes at the top — or go to **Assumptions > Basics** to enter balances manually
3. The simulation re-runs automatically with your actual portfolio

### Saving and Loading Settings

- Click **Save Settings** on the Results page to download a `.json` file with all your current assumptions
- Click **Load Settings** to restore a previously saved configuration
- This lets you compare different scenarios or pick up where you left off

### Pages

- **Results** — main dashboard with success probability, net worth projections, fan chart, and quick sensitivity analysis
- **Assumptions** — all the inputs: ages, portfolio, income, spending, housing, health costs, taxes, market assumptions, and stress tests
- **Deep Dive** — detailed analysis: year-by-year data tables, full sensitivity tornado chart, income decomposition, IRMAA analysis

### Portfolio CSV Format

The app auto-detects CSV exports from most major brokerages. It looks for columns containing dollar values (like "Market Value" or "Current Value") and classifies holdings into asset classes based on their names (e.g., anything with "S&P 500" or "Total Stock" maps to Equities). You can upload CSVs from:

- Schwab
- Fidelity
- Vanguard
- Most other brokerages that let you export positions to CSV

## Troubleshooting

**"streamlit: command not found"**
Python packages were installed but the terminal can't find them. Try:
```bash
python3 -m streamlit run retirement_mc_pro.py
```
(On Windows, use `python -m streamlit run retirement_mc_pro.py`)

**App runs but looks wrong (dark backgrounds, invisible text)**
Make sure the `.streamlit/config.toml` file is present. It forces light mode. If your OS is in dark mode and the config file is missing, things may look broken.

**CSV upload says "files not allowed"**
The app accepts `.csv`, `.txt`, and `.tsv` files. If your brokerage export has a different extension, rename it to `.csv`.

**Port already in use**
If you see an error about port 8501, another instance is probably running. Either close it or specify a different port:
```bash
streamlit run retirement_mc_pro.py --server.port 8502
```
