# My LocoLab Project

基于 LocoLab 和 Isaac Lab 3.0 的独立任务包。初始示例包含 `My-Velocity-Flat-Go2` 和 `My-Velocity-Flat-Go2-Play`，两者沿用 LocoLab 的 Go2 平地环境和 PPO 配置。

## 安装

在本机复用已装有 Isaac Lab、LocoLab 和 Z-RL 的 Python 3.12 环境。以下命令使用该环境现有的 setuptools（需 64 或以上）构建 editable 安装，并保留已有依赖版本。

```bash
cd /home/kang/external_project
source /home/kang/env_isaaclab3/.venv/bin/activate
uv pip install --no-deps --no-build-isolation -e source/my_locolab_project
```

`pyproject.toml` 声明了运行依赖。若迁移到别的机器，先在目标环境中按 LocoLab 的安装说明配置匹配的上游版本，再安装本项目。[LocoLab](https://github.com/syw-robotics/LocoLab)

## 目录结构

```text
external_project/
├── .git/                          # 本项目的 Git 元数据
├── .gitignore                     # 忽略日志、模型、缓存和本地环境
├── README.md                      # 安装、运行和维护说明
├── scripts/z_rl/
│   ├── train.py                   # 调用 LocoLab 的 Z-RL 训练入口
│   └── play.py                    # 调用 LocoLab 的 Z-RL 回放入口
└── source/my_locolab_project/
    ├── pyproject.toml             # 构建、依赖、包发现和插件元数据
    └── my_locolab_project/
        ├── __init__.py            # 包入口
        └── tasks/
            ├── __init__.py        # LocoLab 加载的任务入口
            └── locomotion/
                ├── __init__.py    # 运动控制任务的包边界
                └── go2/
                    └── __init__.py  # 注册 Go2 训练和回放任务
```

## 任务如何被发现

Editable 安装将源码路径和 `locolab.tasks` entry point 写入同一个 Python 环境。训练和回放入口会导入 `locolab.tasks`。LocoLab 先注册自带任务，再读取已安装发行包的插件元数据并加载 `my_locolab_project.tasks`。任务入口导入 Go2 注册模块，后者使用 LocoLab 的注册辅助函数把两个任务 ID 加入 Gymnasium。

注册记录指向 Isaac Lab 的 `ManagerBasedRLEnv` 以及 LocoLab 的环境与 agent 配置，实际运行时由 LocoLab 解析并创建环境，再交给 Z-RL runner。将来可以在 `tasks/locomotion/go2/` 中添加继承上游配置的模块，并把新任务的配置路径指向该模块。Reward、Observation、Event 和 Curriculum 等实验逻辑也可以随任务放在这里。

目录组织参考了 [proof-loco 的安装说明](https://github.com/syw-robotics/proof-loco#-installation) 和 [任务包元数据](https://github.com/syw-robotics/proof-loco/blob/main/source/proof_parkour/pyproject.toml)。

## 验证、训练和回放

从仓库根目录运行。以下检查会触发任务注册并列出对应的配置入口，无需启动仿真。

```bash
python -c 'import gymnasium as gym; import locolab.tasks; print(gym.spec("My-Velocity-Flat-Go2")); print(gym.spec("My-Velocity-Flat-Go2-Play"))'
python scripts/z_rl/train.py --task My-Velocity-Flat-Go2 --help
python scripts/z_rl/play.py --task My-Velocity-Flat-Go2 --help
```

训练示例使用 `ovphysx`，环境数量和迭代次数可按需调整。省略 `--viz` 时，当前 Go2 平地任务默认以无界面模式运行。

```bash
python scripts/z_rl/train.py --task My-Velocity-Flat-Go2 \
  --physics ovphysx --num_envs 64 --max_iterations 100 \
  --experiment_name my_go2_flat
```

回放时将路径替换成训练输出的模型文件。当前 LocoLab 的 Z-RL 回放入口会为任务名自动补上 `-Play`。

```bash
python scripts/z_rl/play.py --task My-Velocity-Flat-Go2 \
  --physics ovphysx --num_envs 1 --viser \
  --checkpoint /absolute/path/to/model.pt
```

训练日志写入当前目录的 `logs/z_rl/`。从本仓库运行脚本即可在这里集中管理实验输出。

## VS Code 和 Git

在项目根目录运行 `code .`，或在 VS Code 中打开 `/home/kang/external_project` 文件夹。用 `Python: Select Interpreter` 选择 `/home/kang/env_isaaclab3/.venv/bin/python`。VS Code 的 Source Control 对应本项目的 Git 仓库。

```bash
cd /home/kang/external_project
git status --short --branch
git add .
git diff --cached --stat
git commit -m "Initial external project"
```

## 更新上游和排查兼容性

创建项目时使用的上游源码提交如下。即使包版本号相同，editable checkout 的接口也可能随 Git 提交变化。

| 仓库 | 分支 | 提交 |
| --- | --- | --- |
| IsaacLab | `main` | `7deb4f08df2e` |
| LocoLab | `dev-lab3.0` | `37e567d81ec0` |
| z_rl | `dev/new-spec-tempo` | `4c8f0a2d1b7e` |

分别进入 `/home/kang/IsaacLab`、`/home/kang/LocoLab` 或 `/home/kang/z_rl`，先确认工作区状态并记录 `git rev-parse HEAD`。需要更新该仓库当前追踪分支时，可执行 `git fetch`，查看 `git log --oneline HEAD..@{upstream}` 后执行 `git merge --ff-only '@{upstream}'`。每次更新后回到本项目复查导入、任务注册和两个脚本的 `--help`。若上游改动包元数据或依赖，再根据该版本的安装说明检查对应的 editable 安装和依赖。

- **Git Merge Conflict** 发生在某个仓库合并或变基时。独立仓库让上游更新与个人提交分开管理。可用该仓库的 `git status` 或 `git diff --name-only --diff-filter=U` 找出冲突文件。使用 `--ff-only` 时，Git 会在分支分叉时拒绝更新，应先检查各自独有的提交。
- **API 兼容问题** 可能在三个上游仓库都干净时出现。先用 `sys.executable` 和模块的 `__file__` 核对解释器及源码来源，再根据 traceback 定位缺失的导入、变更的参数或配置字段。任务缺失时，检查 `importlib.metadata.entry_points(group="locolab.tasks")` 和两个 Gym ID。配置加载出错时，检查 `gym.spec(...).kwargs` 指向的环境与 agent 类是否仍存在。对照已知可用提交与当前提交，调整本项目的调用或配置。
