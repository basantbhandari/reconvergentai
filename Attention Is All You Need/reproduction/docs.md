# Python Virtual Environment Setup on Windows 11

This guide explains how to create, activate, and deactivate a Python virtual environment on Windows 11 using Python 3.13.4.

## Prerequisites

- Windows 11
- Python 3.13.4 installed

## Setup Instructions

### 1. Navigate to Your Project Directory

```powershell
cd path\to\your\project
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\activate
```

Once activated, you should see `(.venv)` at the beginning of your command prompt.

### 4. Deactivate the Virtual Environment

When you're done working in the virtual environment:

```powershell
deactivate
```


