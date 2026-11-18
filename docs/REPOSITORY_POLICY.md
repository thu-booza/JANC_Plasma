# Git仓库维护规则

本目录是精简代码仓库；外层工作区是历史数据与分析资料库。只在本目录执行git add/commit/push。

提交前运行：

```bash
python scripts/check_git_content.py
python scripts/verify_baseline.py
```

第一项检查文件大小、数据/二进制类型和私钥头，属于基础防误提交检查，不是完整的密钥扫描。第二项核对冻结基线哈希。

应进入Git：维护中的Python代码、算例配置、机理、版本依赖、小型测试、分析脚本、说明文档、第三方许可证、输入与结果的SHA256清单。

应留在外层本地目录或服务器结果盘：原始NPZ/H5、运行ZIP、日志、大数据集、模型权重、缓存、环境、服务器具体连接资料和SSH密钥。

当前已纳入爆震管相关冻结代码和迁移工具。等离子体源码仍在外层bolsigplus072024-win，待将配置/代码与45 GiB数据拆清后再选择性纳入；不默认发布第三方Windows二进制或文献资料。

冻结baseline是历史参考，不能直接称为V100生产配置。新改动通过维护中的src、scripts、tests模块引入，不悄悄改写旧基线以匹配旧检查点身份。

新结果目录：Windows外层local_results/incoming/<run-id>；分析输出local_results/analysis/<run-id>；服务器~/janc_runs/<run-id>。每个任务记录代码commit、环境版本、配置和输入哈希。

发布目标为 thu-booza/JANC_Plasma，用户指定为公开仓库。仅发布本精简代码目录；推送前必须检查待提交内容。
