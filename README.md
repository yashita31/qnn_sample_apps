# qnn_sample_apps
//README is still a work in progress

**Please note: This has only been validated using Windows 11 Enterprise Snapdragon(R) X Elite**

**To Install:**
   1. git clone repo
   2. Create virtual environment
      ```
      >> python3.11 -m virtual_env env_sample_app_hrnet
      ```
   3. Activate virtual environment
      ```
      >> env_sample_app_hrnet/Scripts/activate.ps1 (Windows: Validated)
      >> src env_sample_app_hrnet/bin/activate (Mac: Validated)
      >> src env_sample_app_hrnet/bin/activate (Linux: Not Validated)  # Will not work via WSL due to camera binding issue within WSL
      ```
   4. Install dependencies
      ```
      >> pip install -r requirements.txt
      ```
   5. Download model from AI Hub
       
      https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation

   6. Transfer model to qnn_sample_apps/models/
      ```
      >> mv Downloads/hrnex_pose.onnx qnn_sample_apps/models/
      ```


**To run:** </br>
```
>> python ./src/hrnet_pose/main.py (from root directory)
>> python ./src/hrnet_pose/main.py --system windows --model hrnet_pose --processor cpu --camera 1 --available_cameras False
```

**Unit testing required prior to pushing to remote repository, run pytest -v from root directory (qnn_sample_apps):**
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
├─ models
│  └─ README.md
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
