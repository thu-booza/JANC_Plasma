# JANC_Plasma

在实验室JANC的代码基础上，加入等离子体模块，用于计算二维爆震管与等离子体计算耦合。

本仓库目前发布精简的爆震管计算基线和Ubuntu迁移工具。等离子体模块尚未整理入库；大型科研数据、完整流场、检查点和本机连接资料不在仓库中。

**当前是迁移与验证阶段，不是已完成验证的V100生产算例。** 双16 GB GPU不会自动合并显存；当前H5自定义AMR入口尚未实现双卡分区。冻结基线仍包含旧版滚动保存逻辑，仅用于追溯，不能据此承诺长期保存可靠性或稳定爆震结果。

## 目录

- `baseline/h5_closed/runtime/`：冻结的Colab长时计算基线，保留源码原始字节。
- `baseline/h5_closed/provenance.json`：逐文件SHA256和基线来源。
- `cases/h5_closed/`：参考配置和3.55 μs完整输入的校验信息；不包含大型输入文件。
- `scripts/`：服务器信息采集、基线核验和小规模GPU环境检查。
- `requirements/`：CUDA12候选依赖；尚不是完整生产环境锁文件。
- `analysis/`、`tests/`：后续可复用分析和回归测试入口；目前仅含目录说明。
- `docs/`：代码与数据分离规则。
- `licenses/`：JANC与JAX-AMR上游许可证，源码版权声明保留。

## 开发与数据管理

Windows修改代码并提交，Ubuntu使用固定版本运行。不要在正在计算的checkout中更新代码。

```text
~/JANC_Plasma/       代码快照或Git checkout
~/janc_envs/         隔离Python环境，不进入Git
~/janc_tools/        用户级工具，不进入Git
~/janc_runs/         输入、检查点、日志、诊断，不进入Git
```

完整场通过单独文件传输并校验；Git只记录输入哈希、配置和代码。SSH私钥、密码、token、环境目录及大型结果不提交。

## 本地检查

```bash
python scripts/check_git_content.py
python scripts/verify_baseline.py
```

第一项是基础误提交检查，不替代完整密钥审计；第二项检查冻结基线是否保持原始字节。

GPU环境验证需要已安装候选依赖，在受控的两GPU环境中执行：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/gpu_smoke.py --out /path/outside/repository/gpu_smoke.json
```

该命令只测试两卡分别执行小规模float64运算及机理加载，不是多GPU AMR验证，也不推进爆震流场。现有检查点数据不随仓库发布，不能仅clone仓库就恢复计算。

## 生产运行前仍需完成

1. 确认CUDA12环境在V100上实际可用，并冻结依赖版本。
2. 测量真实流场显存、时间推进、AMR重构和重启行为。
3. 验证独立进程、服务器磁盘及下载副本的完整恢复，改善旧保存策略。
4. 验证长期统计状态及必要的网格/时间步敏感性。

目前可用完整输入的身份记录为3.55 μs；更晚的6–7 μs完整状态尚未找回。环境测试或短步验证不代表物理结果已收敛。

## 第三方代码

JANC与JAX-AMR对应MIT许可证分别保存在`licenses/`中。保留相关作者声明；本仓库未将第三方Windows程序、文献资料或大型训练数据重新分发。
