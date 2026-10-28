# ContactForge 发布仓库

- 当前发布范围为 V6 三形状笛卡尔 F/T 插入。先读 `README.md` 和 `docs/RELEASE.md`。
- 修改前确认 `pwd`、`git rev-parse --show-toplevel`、`git status --short --branch`。
- Python 包名 `mjlab_contact_prep` 保留用于检查点兼容。核心训练源码与资产的原字节、来源哈希见 `reference/release_manifest.json`。
- A→B 初始化与恢复会检查实现指纹。修改核心源码后须记录新版本和兼容性，不能仅更新哈希来绕过校验。
- GitHub 只发布必要源码、资产、V6 配置、精简文档和直接相关的回归测试。权重、日志、轨迹、冻结运行目录和一次性实验脚本留在本机。
- 原研究仓库、V6 worktree 和本机产物继续保留；不得因发布精简而删除它们。V6 A 保持暂停，不得自动恢复或结束 A、旧 V4/V5 训练。
- 新实验必须使用新输出目录。CI 仅检查源码与来源，不能把它当作新 GPU 训练或成功率复现。
