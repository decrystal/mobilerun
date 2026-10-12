# Android Agent 优化说明

本分支基于 Mobilerun 的 MobileAgent → FastAgent → ToolRegistry → Android driver，加入通用 Jev-first 决策与证据闭环。每步优先使用原生控件和 Jev，GLM 按需处理文字参数、视觉歧义、恢复和完整目标复核。启动时编译一次任务契约，后续复用。

## 已实现的能力

| 能力 | 实现位置 | 行为 |
| --- | --- | --- |
| Jev 优先和模型回退 | `runtime.py`、`jev.py`、`autoglm.py` | 明确控件操作先走 Jev，其他情况按需升级。 |
| 任务契约与完成判断 | `contracts.py`、`query_scope.py` | 保留数量、范围、过程、保存和最终状态，查询 False/0 也可成功。 |
| 字段身份及输入读回 | `input_verification.py`、`native_text.py`、`native_edit.py`、`native_patch.py` | 文字来源回指、精确输入、字段归属和输入法恢复后的检查。 |
| 动作和保存结果 | `native_proof_binding.py`、`effect_order.py`、`persistent_results.py` | 派发与效果分开，效果未知先查结果，历史输入不能证明保存后的内容。 |
| 跨页查询和计算 | `task_state.py`、`query.py`、`native_query_binding.py` | 保留原生对象/属性/范围，以覆盖证据过滤和计算，绑定真实来源。 |
| 语义进展和有限恢复 | `progress.py`、`input_progress.py`、`runtime.py` | 用实际观察确认子目标进展，为未完成要求生成有界纠正动作。 |
| 有界上下文和视觉历史 | `native_context.py`、`task_state.py`、`autoglm.py` | 共享同控件重复原值，保留实际截图观察与引用，严格限制上传数量。 |

以上模块位于 `mobilerun/agent/decision/`。策略基于用户目标和设备观察，不包含 benchmark 题目或 App 专用执行分支。

空原生树场景按实际截图 SHA256 区分历史分组，解决第三张不同截图到来后第一张图仍有缓存、却被上下文省略的问题。指纹表示捕获的像素，不证明业务身份或语义相同。原生分组、当前动作投影、字符/缓存预算和最多三张审核图片的限制保留。历史图必须实际上传且哈希匹配；纯历史查询可以在后续导航后使用，若复核引用当前图，完成前仍核对该图是否变化。

## 使用

在仓库根目录安装环境：

```powershell
uv sync --frozen --extra jev --no-dev --python 3.11
```

Android 设备需已连接 ADB，并安装兼容的 Mobilerun Portal 和 ADBKeyboard。Jev 使用 `TYPESAFE_API_KEY`，中转 GLM 使用独立的 `GLM_API_KEY`，均从进程环境读取；配置和日志不保存 Key。

```powershell
.\.venv\Scripts\python.exe examples\jev_phone.py `
  --serial emulator-5556 `
  --adb adb `
  --goal '打开设置，查看当前语言，然后告诉我' `
  --fallback-model glm-5.3-flash `
  --reasoning-model glm-5.3-flash `
  --thinking disabled `
  --fallback-base-url 'http://66.154.100.178:8080/v1'
```

`--goal` 为自然语言任务。`--success-package`、`--success-text` 和 `--text` 均可省略；成功断言是附加条件，不能代替完整目标判断。指定自己的设备序列号；`--help` 可在设备离线时查看参数。

默认动作和复核模型都是 `glm-5.3-flash`，thinking disabled，单次请求期限 45 秒。运行结果包含显式 `final_answer`，决策、证据、耗时与恢复过程可由 CLI 输出追踪。

## 代码验证与边界

本版选定的 Agent 行为与回归检查共 1310 项通过，Ruff、运行时导入和 CLI 入口通过。最后的视觉历史改动新增 20 个控制，覆盖三/四帧保留、窗口省略、原生兼容、实际运行时上传、缺图/错图拒绝和当前帧复核。

按用户要求，本轮未新增真实模型、设备或 benchmark 调用，实际成功率、性能和提分尚未重新测量。文件导出、跨进程持久续跑、独立 OCR 和 iOS 是后续扩展。
