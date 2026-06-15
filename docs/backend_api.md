# UFM HDF5 Backend API and Library Guide

本文档说明当前后端代码结构、HTTP 接口、运算层函数，以及如何作为 Python 库直接调用 HDF5 解析和动画生成能力。

## 1. 后端目录结构

```text
backend/
  app.py
  api/
    routes.py
    schemas.py
  services/
    animation.py
    files.py
    hdf5_inspector.py
    task_manager.py
  storage/
    uploads/
    outputs/
```

### 1.1 入口层

`backend/app.py`

- 创建 FastAPI 应用。
- 注册 CORS。
- 注册 `/api` 路由。
- 挂载生成结果目录 `/outputs`。
- 提供前端入口页面和前端静态资源访问。

### 1.2 接口层

`backend/api/routes.py`

接口层只做 HTTP 请求处理，不直接写 HDF5 解析或动画计算逻辑。

主要职责：

- 接收上传文件。
- 调用 `services` 层函数。
- 创建后台任务。
- 查询任务进度。
- 下载生成结果。

`backend/api/schemas.py`

定义 HTTP 请求和响应模型：

- `AnimationRequest`
- `TaskResponse`
- `TaskStatus`

### 1.3 运算层 / 库函数层

`backend/services/hdf5_inspector.py`

负责 HDF5 文件结构解析，主要用于获取：

- `Fracture Simulation N` group 列表。
- 每个 group 的显示名称 `Name`。
- Element 级别属性列表。
- Cell 级别属性列表。

`backend/services/animation.py`

负责动画生成，主要包含：

- 读取 Element 级别数据。
- 读取 Cell 级别数据。
- 将 Element 切分为 Cell 面片。
- 构建 Matplotlib 3D 动画。
- 保存 GIF 或 MP4。

`backend/services/files.py`

负责上传文件和输出文件的存储路径管理。

`backend/services/task_manager.py`

负责后台任务管理和进度状态维护。

## 2. HTTP API 文档

默认后端地址：

```text
http://127.0.0.1:8090
```

如果使用 Nginx 代理，前端通常通过同源地址访问：

```text
/api/...
```

### 2.0 列出系统已有 H5 文件

```http
GET /api/files/h5
```

作用：

- 列出 `data/hdf5_file` 下已有的 H5 文件。
- 列出 `backend/storage/uploads/h5` 下用户上传过的 H5 文件。
- 返回的 `file_id` 可直接用于解析、生成动画和最终状态预览。

返回示例：

```json
{
  "files": [
    {
      "file_id": "data_h5:JY68-4HF.h5",
      "filename": "JY68-4HF.h5",
      "source": "data_h5",
      "size": 123456789
    },
    {
      "file_id": "upload_h5:new_case.h5",
      "filename": "new_case.h5",
      "source": "upload_h5",
      "size": 123456789
    }
  ]
}
```

### 2.0.1 列出系统已有井轨迹文件

```http
GET /api/files/well
```

作用：

- 列出 `data/well_data` 下已有的井轨迹文件。
- 列出 `backend/storage/uploads/well` 下用户上传过的井轨迹文件。

返回示例：

```json
{
  "files": [
    {
      "file_id": "data_well:68-4HF",
      "filename": "68-4HF",
      "source": "data_well",
      "size": 124493
    }
  ]
}
```

### 2.0.2 解析系统已有 H5 文件

```http
POST /api/files/inspect-h5
Content-Type: application/json
```

请求体：

```json
{
  "file_id": "data_h5:JY68-4HF.h5"
}
```

返回结构与上传 H5 后的解析结果一致。

### 2.1 上传并解析 H5 文件

```http
POST /api/files/upload-h5
Content-Type: multipart/form-data
```

请求字段：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `file` | file | 是 | `.h5` 或 `.hdf5` 文件 |

示例：

```bash
curl -X POST http://127.0.0.1:8090/api/files/upload-h5 \
  -F "file=@D:/git/ufm-hdf5/data/hdf5_file/JY68-4HF.h5"
```

返回示例：

```json
{
  "file_id": "bcc1dbe3e6554672a8ccac54d924b9be",
  "filename": "JY68-4HF.h5",
  "groups": [
    {
      "index": 0,
      "group": "Fracture Simulation 45",
      "name": "JY68-4HF - Proposed completion 1 - Stage 1 - Pumping schedule 1 - Treatment design 1",
      "simulation_id": 45,
      "time_steps": 84,
      "element_count": 227,
      "has_geometry": true
    }
  ],
  "element_properties": [
    "AvgPropWidth",
    "AvgWidth",
    "FluidPressure",
    "NetPressure"
  ],
  "cell_properties": [
    "WidthProfile",
    "ProppedWidthProfile",
    "FractureConductivity"
  ]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `file_id` | 后端保存上传文件后生成的 ID，后续生成动画时使用 |
| `groups` | 可选择展示的 UFM group 列表 |
| `group` | HDF5 中真实 group 名称，例如 `Fracture Simulation 45` |
| `name` | group 对应的显示名称，从 HDF5 中的 `Name` 信息读取 |
| `simulation_id` | group 名称末尾的数字 ID |
| `time_steps` | 当前 group 的时间步数量 |
| `element_count` | 当前 group 的最大 element 数量 |
| `element_properties` | Element 级别可展示属性 |
| `cell_properties` | Cell 级别可展示属性 |

说明：

- 上传后的 H5 文件会保留原始文件名，保存到 `backend/storage/uploads/h5`。
- 返回的 `file_id` 形如 `upload_h5:filename.h5`。
- 上传完成后，该文件也会出现在 `/api/files/h5` 列表中。

### 2.2 上传井轨迹文件

```http
POST /api/files/upload-well
Content-Type: multipart/form-data
```

请求字段：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `file` | file | 是 | 井轨迹文件，通常包含 `X`、`Y`、`TVD` 列 |

示例：

```bash
curl -X POST http://127.0.0.1:8090/api/files/upload-well \
  -F "file=@D:/git/ufm-hdf5/data/well_data/68-4HF"
```

返回示例：

```json
{
  "file_id": "1ad44703f1654b4591a405981a7afc1e",
  "filename": "68-4HF"
}
```

`well_file_id` 是可选的。如果生成动画时不传井轨迹文件，动画只展示裂缝，不叠加井轨迹。

说明：

- 上传后的井轨迹文件会保留原始文件名，保存到 `backend/storage/uploads/well`。
- 返回的 `file_id` 形如 `upload_well:filename`。
- 系统自带井轨迹文件的 `file_id` 形如 `data_well:68-4HF`。

### 2.3 创建动画任务

```http
POST /api/animations
Content-Type: application/json
```

请求体：

```json
{
  "file_id": "bcc1dbe3e6554672a8ccac54d924b9be",
  "group_names": [
    "Fracture Simulation 45",
    "Fracture Simulation 46"
  ],
  "group_display_names": {
    "Fracture Simulation 45": "JY68-4HF - Proposed completion 1 - Stage 1 - Pumping schedule 1 - Treatment design 1",
    "Fracture Simulation 46": "JY68-4HF - Proposed completion 1 - Stage 2 - Pumping schedule 1 - Treatment design 1"
  },
  "level": "element",
  "property_name": "NetPressure",
  "renderer": "matplotlib",
  "output_format": "gif",
  "well_file_id": null,
  "fps": 20,
  "interval": 50,
  "keep_aspect": true,
  "time_step_stride": 4
}
```

参数说明：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `file_id` | string | 是 | 无 | 上传 H5 后返回的文件 ID |
| `group_names` | string[] | 是 | 无 | 用户选择的 group 名称 |
| `group_display_names` | object/null | 否 | `null` | group 到 Name 的映射；动画标题会从 Name 中提取 `Stage N` |
| `level` | string | 是 | 无 | `element` 或 `cell` |
| `property_name` | string | 是 | 无 | 要展示的属性名 |
| `renderer` | string | 否 | `matplotlib` | 渲染器，支持 `matplotlib` 和 `pyvista` |
| `output_format` | string | 否 | `gif` | `gif` 或 `mp4` |
| `well_file_id` | string/null | 否 | `null` | 上传井轨迹后返回的文件 ID |
| `fps` | int | 否 | `20` | 输出动画帧率，范围 1-60 |
| `interval` | int | 否 | `50` | Matplotlib 动画帧间隔，单位 ms |
| `keep_aspect` | bool | 否 | `true` | 是否保持 3D 坐标轴比例 |
| `time_step_stride` | int | 否 | `1` | 时间步采样间隔；`1` 表示全部时间步，`2/4/10` 表示每 2/4/10 个时间步取一帧，且每个 group 的最后时间步一定保留 |

PyVista/VTK 渲染器说明：

- 请求体中设置 `"renderer": "pyvista"` 会使用 PyVista/VTK 后端生成动画。
- 需要额外安装依赖：

```bash
pip install pyvista vtk imageio imageio-ffmpeg
```

- 该渲染器使用 VTK/OpenGL/EGL，而不是 CUDA。服务器即使有 H100，也需要 NVIDIA 驱动提供可用的 OpenGL/EGL 渲染环境。

返回示例：

```json
{
  "task_id": "7ffef2c113a64dcfbcdb2ff6389971ef"
}
```

说明：

- 创建动画是耗时操作，因此接口只返回 `task_id`。
- 前端或调用方需要使用任务查询接口轮询进度。

### 2.4 查询任务进度

```http
GET /api/tasks/{task_id}
```

返回示例：

```json
{
  "task_id": "7ffef2c113a64dcfbcdb2ff6389971ef",
  "status": "running",
  "percent": 52,
  "current_group": "Fracture Simulation 46",
  "current_group_index": 2,
  "total_groups": 2,
  "message": "Rendering Fracture Simulation 46 frame 12/45.",
  "result_url": null,
  "error": null
}
```

`status` 可能值：

| 状态 | 说明 |
|---|---|
| `pending` | 任务已创建，等待执行 |
| `running` | 正在读取 HDF5 或渲染动画 |
| `completed` | 动画生成完成 |
| `failed` | 任务失败 |

完成时返回：

```json
{
  "status": "completed",
  "percent": 100,
  "result_url": "/api/tasks/7ffef2c113a64dcfbcdb2ff6389971ef/result"
}
```

### 2.5 下载动画结果

```http
GET /api/tasks/{task_id}/result
```

返回：

- GIF 文件，或
- MP4 文件

前端可以直接把该地址放入：

```html
<img src="/api/tasks/{task_id}/result">
```

或：

```html
<video src="/api/tasks/{task_id}/result" controls></video>
```

## 3. HDF5 解析库 API

库函数位于：

```python
backend.services.hdf5_inspector
```

### 3.1 inspect_hdf5_file

```python
from backend.services.hdf5_inspector import inspect_hdf5_file

metadata = inspect_hdf5_file("data/hdf5_file/JY68-4HF.h5")
```

函数签名：

```python
def inspect_hdf5_file(h5_file_path: str | Path) -> dict[str, Any]
```

作用：

- 打开 HDF5 文件。
- 查找所有顶层 `Fracture Simulation N` group。
- 按 `N` 从小到大排序。
- 读取每个 group 的显示名称 `Name`。
- 统计每个 group 的时间步和 element 数量。
- 收集 Element 级别属性。
- 收集 Cell 级别属性。

参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `h5_file_path` | `str | Path` | HDF5 文件路径 |

返回：

```python
{
    "groups": [...],
    "element_properties": [...],
    "cell_properties": [...]
}
```

调用示例：

```python
from backend.services.hdf5_inspector import inspect_hdf5_file

metadata = inspect_hdf5_file("data/hdf5_file/JY68-4HF.h5")

for group in metadata["groups"]:
    print(group["group"], group["name"])

print("Element properties:", metadata["element_properties"])
print("Cell properties:", metadata["cell_properties"])
```

### 3.2 list_fracture_simulation_groups

```python
import h5py
from backend.services.hdf5_inspector import list_fracture_simulation_groups

with h5py.File("data/hdf5_file/JY68-4HF.h5", "r") as h5_file:
    groups = list_fracture_simulation_groups(h5_file)
```

函数签名：

```python
def list_fracture_simulation_groups(h5_file: h5py.File) -> list[str]
```

作用：

- 从 HDF5 顶层 group 中查找符合 `Fracture Simulation N` 格式的 group。
- 按末尾数字 `N` 排序。

参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `h5_file` | `h5py.File` | 已打开的 HDF5 文件对象 |

返回示例：

```python
[
    "Fracture Simulation 45",
    "Fracture Simulation 46",
    "Fracture Simulation 47"
]
```

### 3.3 read_group_display_name

```python
import h5py
from backend.services.hdf5_inspector import read_group_display_name

with h5py.File("data/hdf5_file/JY68-4HF.h5", "r") as h5_file:
    name = read_group_display_name(h5_file["Fracture Simulation 45"])
```

函数签名：

```python
def read_group_display_name(group: h5py.Group) -> str
```

作用：

- 读取 group 的显示名称。
- 优先读取 group attribute 中的 `Name` 或 `name`。
- 如果 attribute 没有，则查找直接子节点 `Name` 或 `name`。
- 如果还没有，则递归查找路径末尾为 `Name` 的 dataset。
- 如果全部找不到，则返回 group 自身名称。

### 3.4 list_dataset_names

```python
from backend.services.hdf5_inspector import list_dataset_names
```

函数签名：

```python
def list_dataset_names(group: h5py.Group) -> list[str]
```

作用：

- 返回某个 HDF5 group 直接下级的 dataset 名称。
- 不递归读取子 group。

用途：

- 用于读取 `Elements` 下面的 Element 属性。
- 用于读取 `Elements/Cells` 下面的 Cell 属性。

### 3.5 clean_invalid

```python
from backend.services.hdf5_inspector import clean_invalid
```

函数签名：

```python
def clean_invalid(values: np.ndarray) -> np.ndarray
```

作用：

- 将输入转换为 `float` 数组。
- 将非有限值和 UFM 哨兵值替换为 `np.nan`。
- 当前哨兵阈值为：

```python
INVALID_LIMIT = 1.0e30
```

### 3.6 safe_values

```python
from backend.services.hdf5_inspector import safe_values
```

函数签名：

```python
def safe_values(values: np.ndarray) -> np.ndarray
```

作用：

- 返回有效数值。
- 会过滤掉：
  - `nan`
  - `inf`
  - 绝对值大于等于 `1.0e30` 的哨兵值

用途：

- 计算坐标范围。
- 计算属性颜色范围。

## 4. 动画生成库 API

库函数位于：

```python
backend.services.animation
```

### 4.1 generate_ufm_animation

```python
from backend.services.animation import generate_ufm_animation

output_path = generate_ufm_animation(
    h5_file_path="data/hdf5_file/JY68-4HF.h5",
    group_names=["Fracture Simulation 45"],
    level="element",
    property_name="NetPressure",
    output_dir="backend/storage/outputs",
    output_format="gif",
)
print(output_path)
```

函数签名：

```python
def generate_ufm_animation(
    h5_file_path: str | Path,
    group_names: list[str],
    level: Literal["element", "cell"],
    property_name: str,
    output_dir: str | Path,
    output_format: Literal["gif", "mp4"] = "gif",
    well_file_path: str | Path | None = None,
    fps: int = 20,
    interval: int = 50,
    keep_aspect: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Path
```

作用：

- 读取指定 HDF5 文件。
- 根据 `group_names` 加载用户选择的 group。
- 根据 `level` 选择 Element 或 Cell 数据读取方式。
- 根据 `property_name` 读取对应属性值，并映射为颜色。
- 构建 Matplotlib 3D 动画。
- 输出 GIF 或 MP4 文件。

参数说明：

| 参数 | 类型 | 说明 |
|---|---|---|
| `h5_file_path` | `str | Path` | HDF5 文件路径 |
| `group_names` | `list[str]` | 要展示的 group 名称列表 |
| `level` | `"element" | "cell"` | 展示级别 |
| `property_name` | `str` | 属性名称 |
| `output_dir` | `str | Path` | 输出目录 |
| `output_format` | `"gif" | "mp4"` | 输出格式 |
| `well_file_path` | `str | Path | None` | 可选井轨迹文件路径 |
| `fps` | `int` | 保存动画的帧率 |
| `interval` | `int` | Matplotlib 帧间隔，单位 ms |
| `keep_aspect` | `bool` | 是否保持 3D 坐标轴比例 |
| `progress_callback` | callable/null | 进度回调函数 |

Element 示例：

```python
from backend.services.animation import generate_ufm_animation

output_path = generate_ufm_animation(
    h5_file_path="data/hdf5_file/JY68-4HF.h5",
    group_names=[
        "Fracture Simulation 45",
        "Fracture Simulation 46",
    ],
    level="element",
    property_name="NetPressure",
    output_dir="output",
    output_format="gif",
    fps=20,
)
```

Cell 示例：

```python
from backend.services.animation import generate_ufm_animation

output_path = generate_ufm_animation(
    h5_file_path="data/hdf5_file/JY68-4HF.h5",
    group_names=["Fracture Simulation 45"],
    level="cell",
    property_name="WidthProfile",
    output_dir="output",
    output_format="gif",
)
```

带井轨迹示例：

```python
from backend.services.animation import generate_ufm_animation

output_path = generate_ufm_animation(
    h5_file_path="data/hdf5_file/JY68-4HF.h5",
    group_names=["Fracture Simulation 45"],
    level="element",
    property_name="NetPressure",
    output_dir="output",
    well_file_path="data/well_data/68-4HF",
)
```

带进度回调示例：

```python
from backend.services.animation import generate_ufm_animation

def progress(**kwargs):
    print(kwargs)

output_path = generate_ufm_animation(
    h5_file_path="data/hdf5_file/JY68-4HF.h5",
    group_names=["Fracture Simulation 45"],
    level="element",
    property_name="NetPressure",
    output_dir="output",
    progress_callback=progress,
)
```

回调参数可能包含：

```python
{
    "percent": 42,
    "current_group": "Fracture Simulation 45",
    "current_group_index": 1,
    "total_groups": 1,
    "message": "Rendering Fracture Simulation 45 frame 20/84."
}
```

### 4.2 read_element_stage_data

```python
def read_element_stage_data(
    h5_file: h5py.File,
    group_name: str,
    property_name: str,
) -> dict
```

作用：

- 读取某个 group 的 Element 级别数据。
- 读取路径：

```text
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/X
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/Y
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/Z
{group}/Results/Bulk/UFM/FractureSet/Elements/{property_name}
```

返回结构：

```python
{
    "id": group_name,
    "X": ndarray,
    "Y": ndarray,
    "Z": ndarray,
    "Property": ndarray,
    "n_time": int,
    "n_element_max": int,
}
```

数组含义：

| 字段 | 形状 | 说明 |
|---|---|---|
| `X/Y/Z` | `time x element x point` | element 四边形点坐标 |
| `Property` | `time x element` | Element 级别属性值 |

### 4.3 read_cell_stage_data

```python
def read_cell_stage_data(
    h5_file: h5py.File,
    group_name: str,
    property_name: str,
) -> dict
```

作用：

- 读取某个 group 的 Cell 级别数据。
- 几何仍来自 Element 的四边形点坐标。
- 属性来自 `Elements/Cells/{property_name}`。

读取路径：

```text
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/X
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/Y
{group}/Results/Bulk/UFM/FractureSet/Elements/Points/Z
{group}/Results/Bulk/UFM/FractureSet/Elements/Cells/{property_name}
```

返回结构：

```python
{
    "id": group_name,
    "X": ndarray,
    "Y": ndarray,
    "Z": ndarray,
    "Property": ndarray,
    "ElementCount": ndarray,
    "n_time": int,
    "n_element_max": int,
    "n_cells": int,
}
```

数组含义：

| 字段 | 形状 | 说明 |
|---|---|---|
| `X/Y/Z` | `time x element x point` | element 四边形点坐标 |
| `Property` | `time x element x cell` | Cell 级别属性值 |
| `ElementCount` | `time` | 每个时间步的有效 element 数 |

### 4.4 split_element_to_cell_quads

```python
def split_element_to_cell_quads(
    points4: np.ndarray,
    split_count: int,
) -> list[np.ndarray]
```

作用：

- 将一个 element 四边形切分为多个 cell 四边形。
- `split_count` 必须来自 HDF5 中 Cell 属性数组的最后一维。
- 例如 `WidthProfile.shape == (84, 227, 31)` 时，每个 element 应拆成 31 个 cell 面片。
- Cell 级别动画计算量明显大于 Element 级别，因为每个 element 会被拆成 `Property.shape[2]` 个面片。

参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `points4` | `np.ndarray` | 一个 element 的 4 个点坐标，形状通常是 `4 x 3` |
| `split_count` | `int` | 切分片数 |

### 4.5 build_element_verts_for_stage

```python
def build_element_verts_for_stage(
    sim_data: dict,
    time_step: int,
    cmap,
    norm,
    alpha: float,
) -> tuple[list, list]
```

作用：

- 将某个时间步的 Element 数据转换成 Matplotlib `Poly3DCollection` 可用的面片数据。
- 返回：
  - `verts`: 四边形顶点列表。
  - `face_colors`: 每个面片对应颜色。

### 4.6 build_cell_verts_for_stage

```python
def build_cell_verts_for_stage(
    sim_data: dict,
    time_step: int,
    cmap,
    norm,
    alpha: float,
) -> tuple[list, list]
```

作用：

- 将某个时间步的 Cell 数据转换成 Matplotlib 3D 面片。
- 每个有效 Element 会被 `split_element_to_cell_quads` 切成多个 cell 面片。
- 使用 Cell 属性值给每个 cell 面片上色。
- 切分数量来自 `sim_data["n_cells"]`，也就是 Cell 属性数组最后一维，不由前端或调用方手动传入。

## 5. HDF5 数据路径约定

当前代码默认 UFM HDF5 文件结构如下。

### 5.1 Group 命名

```text
Fracture Simulation 45
Fracture Simulation 46
...
```

代码只识别符合以下正则的顶层 group：

```text
^Fracture Simulation\s+(\d+)$
```

### 5.2 Element 级别路径

```text
Fracture Simulation N/
  Results/
    Bulk/
      UFM/
        FractureSet/
          Elements/
            Points/
              X
              Y
              Z
            NetPressure
            Temperature
            FluidPressure
            ...
```

Element 属性直接位于：

```text
Elements/{property_name}
```

### 5.3 Cell 级别路径

```text
Fracture Simulation N/
  Results/
    Bulk/
      UFM/
        FractureSet/
          Elements/
            Cells/
              WidthProfile
              ProppedWidthProfile
              FractureConductivity
              ...
```

Cell 属性位于：

```text
Elements/Cells/{property_name}
```

## 6. 调用流程建议

### 6.1 Web 调用流程

1. 调用 `/api/files/upload-h5` 上传 H5。
2. 根据返回的 `groups`、`element_properties`、`cell_properties` 让用户选择。
3. 可选调用 `/api/files/upload-well` 上传井轨迹。
4. 调用 `/api/animations` 创建动画任务。
5. 轮询 `/api/tasks/{task_id}` 获取进度。
6. 完成后访问 `/api/tasks/{task_id}/result` 下载或预览动画。

### 6.2 Python 库调用流程

```python
from backend.services.hdf5_inspector import inspect_hdf5_file
from backend.services.animation import generate_ufm_animation

h5_path = "data/hdf5_file/JY68-4HF.h5"

metadata = inspect_hdf5_file(h5_path)
groups = [metadata["groups"][0]["group"]]

output_path = generate_ufm_animation(
    h5_file_path=h5_path,
    group_names=groups,
    level="element",
    property_name="NetPressure",
    output_dir="output",
    output_format="gif",
)

print(output_path)
```

## 7. 注意事项

- H5 文件通常较大，Nginx 需要配置 `client_max_body_size`。
- `/api/` 必须反向代理到 FastAPI，否则上传会被 Nginx 静态服务拒绝。
- Cell 级别计算量远大于 Element 级别，因为每个 element 会切成 HDF5 Cell 属性最后一维对应的面片数量。
- MP4 输出依赖本机可用的 `ffmpeg` writer。
- GIF 输出依赖 `pillow`。
- 当前任务管理器是内存态的，服务重启后任务状态会丢失。
- 当前上传文件和生成结果保存在 `backend/storage`，该目录默认不提交到 Git。
