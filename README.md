# qnn_sample_apps
//README is still a work in progress

**<u>Please note: This has only been validated using Windows 11 Enterprise Snapdragon(R) X Elite</u>**

**To Install:**
   1. git clone repo
   2. Create virtual environment
      ```
      >> python3.11 -m virtual_env env_sample_app_hrnet
      ```
   3. Activate virtual environment
      ```
      >> env_sample_app_hrnet/Scripts/activate.ps1 (Windows)
      >> src env_sample_app_hrnet/bin/activate (Linux)  #hasn't been tested in Linux, won't work via WSL
      ```
   4. Install dependencies
      ```
      >> pip install -r requirements.txt
      ```


**To run:** </br>
```
>> python ./src/hrnet_pose/main.py (from root directory)
>> python ./src/hrnet_pose/main.py --system "windows" --model "hrnet_pose" --processor "cpu" --available_cameras False
```

**Before making any push run pytest -v from root directory (qnn_sample_apps)**
```
>> cd \qnn_sample_apps
>> pytest -v (-vv)
```
**Directory Structure:**
```
qnn_sample_apps
├─ .gitignore
├─ dll.json
├─ executioner.json
├─ models.json
├─ notebooks
│  └─ sample_app_hrnet.ipynb
├─ pyproject.toml
├─ README.md
├─ requirements.in
├─ requirements.txt
├─ scripts
│  └─ directory_information.txt
├─ setup.py
├─ src
│  ├─ hrnet_pose
│  │  ├─ main.py
│  │  ├─ model_inference.py
│  │  ├─ model_loader.py
│  │  └─ README.md
│  └─ __init__.py
└─ tests
   └─ test_module.py

```