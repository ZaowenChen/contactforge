# ContactForge 发布范围与来源

2026-10-28，项目以 **ContactForge（接触反馈机器人装配）** 建立新的精简发布仓库。README 参考用户维护的《MPHS 项目总览｜V6三形状插入与下一阶段探索》（2026-10-26），方法与数值对照实际源码、配置和结果清单。Obsidian 私有链接、本机绝对路径及未验证的因果解释未直接搬入 README。

## 源码来源

- 原仓库：[ZaowenChen/mjlab-peg-hole-search](https://github.com/ZaowenChen/mjlab-peg-hole-search)。
- 整合提交：[`830fc40f1588902658d0343b1e69da22561b4173`](https://github.com/ZaowenChen/mjlab-peg-hole-search/commit/830fc40f1588902658d0343b1e69da22561b4173)。
- 算法基线：已完成 B1000 的冻结实现。核心 Python 模块沿用原字节；`seating.py` 恢复训练快照中的末尾空行，AST 不变，使旧检查点的严格实现指纹重新匹配。
- 源码与资产校验清单：[release_manifest.json](../reference/release_manifest.json)。
- 实验摘要：[v6_b1000.json](../reference/v6_b1000.json)。

新仓库使用独立 Git 历史。原仓库及其分支、worktree、冻结源码、权重、日志和轨迹继续保留。新发布没有切换旧仓库远程，也没有启动、恢复或结束任何历史训练。

## 精简范围

| 部分 | 原仓库 | 本发布 |
|---|---:|---:|
| Python 运行源码 | 66 个文件 | 28 个文件 |
| `scripts/` 中的 Python 入口 | 34 个文件 | 3 个文件 |
| 测试文件 | 23 个文件 | 2 个文件 |
| 资产文件 | 108 个文件 | 26 个文件 |
| 配置 JSON | 16 个文件 | 2 个文件 |

发布范围由导入依赖、模型 XML 引用和检查点指纹确定，具体数量以清单为准。

保留 V6 训练、评估、单阶段 / 课程启动、控制器、网络、必要基类、F/T 修正和碰撞兼容层。模型只保留机械臂、夹爪和轴引用的 XML / 网格；V6 孔模型按配置生成。

未复制到新仓库的内容包括旧 SPIKE 入口、无关控制 / 搜索管线、历史配置、一次性绘图与诊断脚本、旧孔网格、旧报告集合和无关测试。原文件仍在原仓库。

以下文件虽保留旧命名或兼容分支，仍有明确用途：

- `environment.py`、`study_environment.py`、`polygon_environment.py`、`control.py`、`networks.py` 是 V6 的继承与公共实现。
- `clearance_validation.py` 约束 A→B 的资产与初始化合同。
- `cartesian_validation.py` 支持 V6 控制诊断，也属于训练指纹。
- `seating_ablation.py` 被检查点评估的兼容校验引用，也属于训练指纹。
- 根包的 `assets.py`、`backend.py`、`hole_collision.py`、`wrench_torch.py` 属于运行依赖及实现指纹。根资产适配器中的历史场景辅助函数不属于本发布支持入口。

不改 Python 包名、不重写已验证控制或奖励，有助于保留旧权重来源关系。运行器和项目元数据已更新为 ContactForge；源码路径与模型键保持原兼容形式。

原 A100 配置保留分段评估计划，而新默认仅最终评估。严格初始化合同也会比较该字段；[原 A100 兼容命令](V6.md#使用原实验的-a100) 使用本地配置保留元数据，再显式覆盖作业评估节点，不修改训练算法或绕过检查。

## 保留哪些测试

`test_v6_cartesian.py` 检查 A/B 合同、倒角与座深、过载优先级、卸载方向、旋转运算、Actor 特权输入隔离及辅助损失。

`test_v6_wrench_sensor.py` 检查交错接触约束行的 F/T 解码，避免把其他接触的约束力混入当前读数。这两份测试直接保护 V6 行为，保留在发布中。

## 不随 Git 发布的产物

`.gitignore` 排除 `evaluation/`、`runs/`、检查点、二进制模型、数组轨迹、TensorBoard 事件、日志、冻结压缩包和本机发布核查目录。保留可审阅的结果摘要、一张训练曲线、许可和哈希清单。新源码或新配置必须使用新实验目录。

## 方法来源

历史编码、接触辅助监督和非对称 Critic 的参考与本地适配记录见原仓库的 [公开来源清单](https://github.com/ZaowenChen/mjlab-peg-hole-search/blob/830fc40f1588902658d0343b1e69da22561b4173/reference/blind_study_public_sources_20261020.json)。V6 增加 F/T 观测、双流编码、笛卡尔执行、位姿辅助监督和本项目成功协议，不将结果表述为论文原方法的同条件复现。

MuJoCo Warp 兼容代码的上游版本、修改和 Apache-2.0 许可见 [_vendor/README.md](../src/mjlab_contact_prep/_vendor/README.md)。机器人和 CAD 资产保留原字节与原权属说明，见 [ASSETS_AND_THIRD_PARTY.md](../ASSETS_AND_THIRD_PARTY.md)。
