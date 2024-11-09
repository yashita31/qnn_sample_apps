# qnn_sample_apps
//README is still a work in progress

Please note this has only been tested Windows 11 Enterprise  Snapdragon(R) X Elite

To Install:
   1. git clone repo
   2. Create virtual environment
      >> python3.11 -m virtual_env env_sample_app_hrnet
   3. Activate virtual environment
      >> env_sample_app_hrnet/Scripts/activate.ps1 (Windows)
      >> src env_sample_app_hrnet/bin/activate (Linux)  #hasn't been tested in Linux, won't work via WSL
   4. Install dependencies
      >> pip install -r requirements.txt


To run: </br>
python main.py (from root directory) </br>
python main.py --system "windows" --model "hrnet_pose" --processor "cpu" --available_cameras False

**Before making any push run pytest -v from root directory (qnn_sample_apps\)**
>> cd \qnn_sample_apps
>> pytest -v (-vv)


Directory Structure
```
qnn_sample_apps
├─ .gitignore
├─ notebooks
│  └─ sample_app_hrnet.ipynb
├─ pyproject.toml
├─ qnn_sample_apps.egg-info
│  ├─ dependency_links.txt
│  ├─ PKG-INFO
│  ├─ requires.txt
│  ├─ SOURCES.txt
│  └─ top_level.txt
├─ README.md
├─ scripts
│  └─ directory_information.txt
├─ setup.py
├─ src
│  ├─ hrnet_pose
│  │  ├─ main.py
│  │  └─ README.md
│  ├─ __init__.py
│  └─ __pycache__
│     └─ __init__.cpython-311.pyc
└─ tests
   └─ test_module.py

```
```
qnn_sample_apps
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ FETCH_HEAD
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  ├─ sendemail-validate.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ logs
│  │  ├─ HEAD
│  │  └─ refs
│  │     ├─ heads
│  │     │  ├─ develop
│  │     │  └─ main
│  │     └─ remotes
│  │        └─ origin
│  │           ├─ develop
│  │           └─ HEAD
│  ├─ objects
│  │  ├─ 01
│  │  │  └─ e3188c773c07fa4201d394deb34362e1f3610f
│  │  ├─ 07
│  │  │  └─ f10df8a5ac4b72d8290bccb85e5819afc9cf0e
│  │  ├─ 08
│  │  │  └─ c7ee3b84c8b59680ed9a63d1ed8b681cd70382
│  │  ├─ 13
│  │  │  ├─ 7046457d6cdf9de93fbbdd0b5ff3c4e3214dc3
│  │  │  └─ a2a5b71eacd72ffd5d52b294b481e784e64102
│  │  ├─ 16
│  │  │  └─ 906d7b513f3fe57ff23cafa0d39179a1c49340
│  │  ├─ 1d
│  │  │  └─ 773f692b1770f6519fe532d2bc0bd0f56ff776
│  │  ├─ 25
│  │  │  └─ 87724754be2a05c21722b5b50823a1a6bbb1e8
│  │  ├─ 29
│  │  │  ├─ 7c6e5d1189ad52786f620e460968e4cf68de57
│  │  │  └─ d958ee4d70ffc0d961910d500e82c572941e82
│  │  ├─ 2b
│  │  │  └─ cdfd92bacb5e2c34ba4e08b63112abfedaf538
│  │  ├─ 2f
│  │  │  ├─ 9c884fe97196356bcfd8e0f203cb4d36467524
│  │  │  └─ e19e8d1a671d2a331140b7e04b3b997e5020d3
│  │  ├─ 32
│  │  │  └─ bf001e14eac69b5177fcccccf94a185115ab7e
│  │  ├─ 34
│  │  │  └─ 734d23ad5123baddb299bed099efcd4ecbebd2
│  │  ├─ 3a
│  │  │  └─ ff473f8f38bc68ef49dc0693678505dbbc0087
│  │  ├─ 3b
│  │  │  └─ 90dadc89c8c095de0b16a8769ed988dd57c4d5
│  │  ├─ 3d
│  │  │  └─ 3474107268132134826556a2a0a04ee2dba31b
│  │  ├─ 40
│  │  │  ├─ 0543b2fc90c1d690b586884cb9c5488949960f
│  │  │  └─ 0840e74306a1f6c0d6bdcf2b55d436047ee8fa
│  │  ├─ 4b
│  │  │  └─ 8118ad13483311329c822e3cb17e04273f9eeb
│  │  ├─ 50
│  │  │  └─ 4e2b77f2f3895fb9f004e07e3115c40da68dad
│  │  ├─ 57
│  │  │  └─ 2d9ea694dd63165a2dbe684b01554ef262d357
│  │  ├─ 5b
│  │  │  ├─ 3185157c146012c926d695d644055f09ad346c
│  │  │  └─ d5b015b7ff2d8319fe62895c4c158b8e7378ea
│  │  ├─ 69
│  │  │  └─ a5b149a20c88d25fa0b89c4ed9defa6d2ccdf4
│  │  ├─ 6a
│  │  │  ├─ 4f64f5e2e40250943613764792092102712d91
│  │  │  └─ 9ec74cc9931826de77e0cff7026ea418c55697
│  │  ├─ 6d
│  │  │  └─ d50824247c73758ba3dc3f59638feaf62d4be9
│  │  ├─ 6e
│  │  │  ├─ a8874968d000cd47f52f55f32a92f0127532b3
│  │  │  └─ e397f9c533d5d8aa6c4143b39db00e593a0532
│  │  ├─ 71
│  │  │  └─ f5b43c4f49f582e4290a7cb7b6821e510f2c04
│  │  ├─ 7b
│  │  │  └─ 6b06665baf8cd7c4ee5ca26eac86d20d2001ac
│  │  ├─ 8c
│  │  │  └─ 6790b342acc470dddcfde57f455145fa68578a
│  │  ├─ 8e
│  │  │  └─ 4d4608189eea012246b8a21b0cf3511c42ded3
│  │  ├─ 92
│  │  │  └─ bb41f6c2011deeaf0d198e1213fac3bc4836ad
│  │  ├─ 93
│  │  │  └─ f6232e98a6f3462fea9d64f2aa7ad0619e5621
│  │  ├─ 97
│  │  │  └─ 20fdbb57576dbdf2f2194e81076e04335205e2
│  │  ├─ 98
│  │  │  └─ 5ceb6639fbc5424eeeae0b01aa1ce533b0fc66
│  │  ├─ 9b
│  │  │  └─ 3b0bba4eb773cee1caca1993fa39ca54c4c239
│  │  ├─ a1
│  │  │  └─ 5b0a5d8a4463bf52008cb990ecb3dd3eb5facd
│  │  ├─ a2
│  │  │  └─ b1552091068d4f371d6938e37b463bfd6603e3
│  │  ├─ a4
│  │  │  └─ 5bd84501d467dd8ba0c711b6d8bd188eb4fb05
│  │  ├─ a9
│  │  │  └─ ca61636fb5115aa7079a609e117ccb1441c20c
│  │  ├─ ae
│  │  │  └─ 72a446246eea6b8e4cfee0e8448494ebc44b3c
│  │  ├─ b4
│  │  │  └─ ebf4e57eeaa897cd910047662f89b6f4b9a23e
│  │  ├─ b7
│  │  │  └─ 54049c3e74754c7297b29d1b11bcc0165a40f2
│  │  ├─ bb
│  │  │  └─ 0e3edf2ccd9079d123450d6e1684b2574a4daa
│  │  ├─ bc
│  │  │  └─ cca58614d1fb52bc63a7bb8ea4d4930b9d2048
│  │  ├─ bd
│  │  │  └─ 700f8f6c43a50cf80788281901d383dc9e78bf
│  │  ├─ c1
│  │  │  ├─ 7b84d56ef2b95346bff010c376c161f44b2dd6
│  │  │  └─ baade9881e2906107238c7a626e5b69d82e013
│  │  ├─ c3
│  │  │  └─ 9405193a3bb3edfdc908312bc26c3ea6fe426b
│  │  ├─ c9
│  │  │  └─ 8ddb621afed4fd6b474a0d8d40a376cb81317a
│  │  ├─ cc
│  │  │  └─ 50aa9e5f88eaca925c96ee5fbc43ccea9baadc
│  │  ├─ cd
│  │  │  └─ 93f2c241a6add765b0c7f074ba492f441c1f20
│  │  ├─ d0
│  │  │  ├─ 2add0f2339041bc163ab322917ff0f6aed2ced
│  │  │  └─ f8382fa1cb10b4dc0be9a0b0d863d49ae23db0
│  │  ├─ d4
│  │  │  ├─ 9e9748fcd0b5149a4b8157f426fa1cb6cbd587
│  │  │  └─ f8228b3174e3a9f6122260d98a8ca81765b48c
│  │  ├─ d5
│  │  │  └─ de159480e0bfd12a761be834b02e137b3c26df
│  │  ├─ d9
│  │  │  └─ 73b81b559614c28026227c76b622bbd97d70dc
│  │  ├─ df
│  │  │  └─ 4f05f54678f3002c68ef64e8f7da2ed8b05aef
│  │  ├─ e0
│  │  │  └─ d9ba86e3c5123c01092b3629f5ae599bc31e05
│  │  ├─ e6
│  │  │  └─ 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
│  │  ├─ ea
│  │  │  └─ 4d821f55d1e70a31573d36dfd96755c666fa78
│  │  ├─ ec
│  │  │  ├─ f4bc092b7238c7fd5dc54fadcd6eb149f655ea
│  │  │  └─ fdecdfbbdd86dfb42623595aa7495b734dd865
│  │  ├─ ee
│  │  │  └─ d8eb394f74872e7dc406e7efe3af3d0da283c5
│  │  ├─ f0
│  │  │  └─ 6360c62087ec44b49e63f43489d76fd8c6d52d
│  │  ├─ f9
│  │  │  └─ 3e3a1a1525fb5b91020da86e44810c87a2d7bc
│  │  ├─ fd
│  │  │  └─ a491e77dad04151abe7a552d1b479edacbc78d
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-547669d7e09238fc90e6626f96bdd3b1d089db1d.idx
│  │     ├─ pack-547669d7e09238fc90e6626f96bdd3b1d089db1d.pack
│  │     └─ pack-547669d7e09238fc90e6626f96bdd3b1d089db1d.rev
│  ├─ ORIG_HEAD
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  ├─ develop
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ develop
│     │     └─ HEAD
│     └─ tags
├─ .gitignore
├─ .pytest_cache
│  ├─ .gitignore
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        ├─ nodeids
│        └─ stepwise
├─ .vscode
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