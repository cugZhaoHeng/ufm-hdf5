# -*- coding: utf-8 -*-
"""
读取HDF5文件中裂缝信息及Metadata

功能：
1. 读取所有Fracture Simulation（如 Fracture Simulation 03, Fracture Simulation 24）
2. 读取Metadata/Identification下的Name属性
3. 显示裂缝所在的段（Stage）信息
4. 显示每个裂缝的时间步长信息
"""

import h5py
import numpy as np
from typing import Dict, List, Any
import re


def extract_sim_id(sim_name: str) -> int:
    """
    从裂缝名称中提取Simulation ID
    例如: "Fracture Simulation 03" -> 3
         "Fracture Simulation 24" -> 24
    """
    match = re.search(r'Fracture Simulation (\d+)', sim_name)
    if match:
        return int(match.group(1))
    return None


def format_sim_name(sim_id: int) -> str:
    """
    格式化Simulation名称
    例如: 3 -> "Fracture Simulation 03"
         24 -> "Fracture Simulation 24"
    """
    return f"Fracture Simulation {sim_id:02d}"


def get_time_steps_info(file_path: str, sim_path: str) -> Dict[str, Any]:
    """
    获取裂缝的时间步信息
    
    参数:
        file_path: HDF5文件路径
        sim_path: Simulation的完整路径，如 "Fracture Simulation 03"
    
    返回:
        包含时间步信息的字典
    """
    time_info = {
        'n_time_steps': 0,
        'time_values': None,
        'time_unit': 'Unknown',
        'has_time_data': False
    }
    
    with h5py.File(file_path, 'r') as f:
        # 方法1: 从FractureSet/ElementCount获取时间步数
        fracture_set_path = f"{sim_path}/Results/Bulk/UFM/FractureSet"
        if fracture_set_path in f:
            if "ElementCount" in f[fracture_set_path]:
                element_count = f[f"{fracture_set_path}/ElementCount"][:]
                time_info['n_time_steps'] = len(element_count)
                time_info['has_time_data'] = True
        
        # 方法2: 从TimeValues或Time数组获取实际时间值
        time_paths = [
            f"{sim_path}/Results/Bulk/UFM/TimeValues",
            f"{sim_path}/Results/Bulk/UFM/Time",
            f"{sim_path}/Parameters/TimeSteps",
            f"{sim_path}/Metadata/Simulation/TimeValues"
        ]
        
        for time_path in time_paths:
            if time_path in f:
                time_data = f[time_path][:]
                if isinstance(time_data, np.ndarray) and len(time_data) > 0:
                    time_info['time_values'] = time_data
                    # 尝试读取时间单位
                    if "Units" in f[sim_path]:
                        units_path = f"{sim_path}/Units"
                        if "Time" in f[units_path]:
                            time_unit_data = f[f"{units_path}/Time"][()]
                            if isinstance(time_unit_data, bytes):
                                time_info['time_unit'] = time_unit_data.decode('utf-8')
                break
        
        # 方法3: 从数据维度推断时间步
        if time_info['n_time_steps'] == 0:
            # 尝试从Points/X的shape获取
            points_x_path = f"{sim_path}/Results/Bulk/UFM/FractureSet/Elements/Points/X"
            if points_x_path in f:
                x_data = f[points_x_path][:]
                if x_data.ndim >= 1:
                    time_info['n_time_steps'] = x_data.shape[0]
                    time_info['has_time_data'] = True
        
        # 方法4: 从WidthProfile的shape获取
        width_path = f"{sim_path}/Results/Bulk/UFM/FractureSet/Elements/Cells/WidthProfile"
        if width_path in f:
            width_data = f[width_path][:]
            if width_data.ndim >= 1:
                time_info['n_time_steps'] = width_data.shape[0]
                time_info['has_time_data'] = True
    
    return time_info


def list_all_fracture_simulations(file_path: str, show_details: bool = True):
    """
    列出HDF5文件中所有的Fracture Simulation及其详细信息
    
    参数:
        file_path: HDF5文件路径
        show_details: 是否显示详细信息
    """
    print("=" * 120)
    print(f"HDF5 文件: {file_path}")
    print("=" * 120)
    
    # 查找所有Fracture Simulation
    sim_ids = []
    sim_names = {}
    sim_full_names = {}
    sim_time_info = {}
    
    with h5py.File(file_path, 'r') as f:
        # 匹配 "Fracture Simulation XX" 格式，支持带前导零或不带
        pattern = re.compile(r'Fracture Simulation (\d+)')
        
        for key in f.keys():
            match = pattern.match(key)
            if match:
                sim_id = int(match.group(1))
                sim_ids.append(sim_id)
                sim_full_names[sim_id] = key
                
                # 尝试读取Name属性
                name_path = f"{key}/Metadata/Identification/Name"
                if name_path in f:
                    name_data = f[name_path][()]
                    if isinstance(name_data, bytes):
                        sim_names[sim_id] = name_data.decode('utf-8')
                    else:
                        sim_names[sim_id] = str(name_data)
                else:
                    sim_names[sim_id] = f"Unnamed_Sim{sim_id:02d}"
    
    sim_ids.sort()
    
    if not sim_ids:
        print("未找到任何 Fracture Simulation!")
        print("\n提示: 请确保HDF5文件包含形如 'Fracture Simulation XX' 的组")
        print("     其中XX为数字，例如: Fracture Simulation 03, Fracture Simulation 24")
        return
    
    # 获取每个simulation的时间步信息
    print(f"\n正在读取时间步信息...")
    for sim_id in sim_ids:
        full_name = sim_full_names[sim_id]
        sim_time_info[sim_id] = get_time_steps_info(file_path, full_name)
    
    print(f"\n找到 {len(sim_ids)} 个 Fracture Simulation:\n")
    
    # 显示基本信息表格（增加了时间步列）
    print(f"{'ID':<6} {'Full Name':<30} {'Display Name (Metadata)':<40} {'Stage Info':<20} {'Time Steps':<15} {'Time Range':<25}")
    print("-" * 145)
    
    for sim_id in sim_ids:
        full_name = sim_full_names[sim_id]
        display_name = sim_names.get(sim_id, f"Unnamed_Sim{sim_id:02d}")
        # 截断过长的名字
        if len(display_name) > 37:
            display_name = display_name[:37] + "..."
        
        # 获取stage信息
        stage_info = ""
        if show_details:
            try:
                with h5py.File(file_path, 'r') as f:
                    sim_path = full_name
                    # 尝试多种可能的stage信息路径
                    stage_paths = [
                        f"{sim_path}/Results/Bulk/UFM/Mesh/Stage",
                        f"{sim_path}/Parameters/Stages",
                        f"{sim_path}/Metadata/Simulation/StageCount"
                    ]
                    for stage_path in stage_paths:
                        if stage_path in f:
                            stage_data = f[stage_path][()]
                            if isinstance(stage_data, np.ndarray):
                                if stage_data.size > 0:
                                    if stage_data.dtype.kind in ['i', 'u', 'f']:
                                        n_stages = len(np.unique(stage_data))
                                        stage_info = f"{n_stages} stages"
                                        if n_stages == 1:
                                            stage_info = f"{n_stages} stage"
                                    else:
                                        stage_info = str(stage_data[0])[:20]
                            else:
                                stage_info = str(stage_data)[:20]
                            break
            except:
                pass
        
        # 获取时间步信息
        time_info = sim_time_info[sim_id]
        n_steps = time_info['n_time_steps']
        time_steps_str = f"{n_steps} steps" if n_steps > 0 else "N/A"
        
        # 获取时间范围
        time_range_str = ""
        if time_info['time_values'] is not None and len(time_info['time_values']) > 0:
            time_values = time_info['time_values']
            if len(time_values) > 1:
                time_min = np.min(time_values)
                time_max = np.max(time_values)
                time_unit = time_info['time_unit']
                time_range_str = f"{time_min:.2f} - {time_max:.2f} {time_unit}"
            elif len(time_values) == 1:
                time_range_str = f"{time_values[0]:.2f} {time_info['time_unit']}"
        elif n_steps > 0:
            time_range_str = f"0 to {n_steps-1} (index)"
        else:
            time_range_str = "N/A"
        
        print(f"{sim_id:<6} {full_name:<30} {display_name:<40} {stage_info:<20} {time_steps_str:<15} {time_range_str:<25}")
    
    # 显示详细信息
    if show_details:
        print("\n" + "=" * 120)
        print("详细信息 (按裂缝ID):")
        print("=" * 120)
        
        for sim_id in sim_ids:
            full_name = sim_full_names[sim_id]
            print(f"\n{'='*80}")
            print(f"{full_name} (ID: {sim_id})")
            print(f"Metadata Name: {sim_names.get(sim_id, 'Unknown')}")
            print(f"{'='*80}")
            
            # 显示时间步详细信息
            time_info = sim_time_info[sim_id]
            print(f"\n⏱️  时间步信息:")
            print(f"   时间步数: {time_info['n_time_steps']}")
            if time_info['time_values'] is not None:
                print(f"   时间值: {time_info['time_values'][:10]}...")  # 只显示前10个
                if len(time_info['time_values']) > 10:
                    print(f"            ... (共{len(time_info['time_values'])}个)")
            print(f"   时间单位: {time_info['time_unit']}")
            
            # 读取metadata
            metadata = get_simulation_metadata(file_path, full_name)
            if metadata:
                print("\n📋 Metadata/Identification:")
                for key, val in metadata.items():
                    if key == 'Name':
                        print(f"   ⭐ {key}: {val}")
                    else:
                        print(f"   {key}: {val}")
            
            # 读取FractureSet信息
            fracture_info = get_fracture_set_info(file_path, full_name)
            if fracture_info:
                print("\n🔧 FractureSet 信息:")
                for key, val in fracture_info.items():
                    print(f"   {key}: {val}")
            
            # 识别裂缝所在的段（Stage）
            print("\n🎯 裂缝段识别:")
            with h5py.File(file_path, 'r') as f:
                sim_path = full_name
                
                # 方法1: 从Mesh/Stage读取
                mesh_stage_path = f"{sim_path}/Results/Bulk/UFM/Mesh/Stage"
                if mesh_stage_path in f:
                    stage_data = f[mesh_stage_path][:]
                    if isinstance(stage_data, np.ndarray) and stage_data.size > 0:
                        unique_stages = np.unique(stage_data)
                        print(f"   ✅ 从 Mesh/Stage 识别: {len(unique_stages)} 个段")
                        for stage in unique_stages:
                            if isinstance(stage, (int, float)):
                                count = np.sum(stage_data == stage)
                                print(f"      Stage {stage}: {count} 个单元")
                        continue
                
                # 方法2: 从Parameters/Stages读取
                param_stage_path = f"{sim_path}/Parameters/Stages"
                if param_stage_path in f:
                    stages = f[param_stage_path][:]
                    if isinstance(stages, np.ndarray):
                        print(f"   📊 参数中定义的段数: {len(stages)}")
                        for i, stage in enumerate(stages[:5]):  # 最多显示5个
                            if isinstance(stage, bytes):
                                stage = stage.decode('utf-8')
                            print(f"      Stage {i+1}: {stage}")
                        continue
                
                # 方法3: 从Metadata中读取
                meta_stage_path = f"{sim_path}/Metadata/Simulation/StageCount"
                if meta_stage_path in f:
                    stage_count = f[meta_stage_path][()]
                    print(f"   📝 Metadata中记录的段数: {stage_count}")
                    continue
                
                # 方法4: 从裂缝名称推断（如果名称包含Stage信息）
                name_str = sim_names.get(sim_id, '')
                stage_match = re.search(r'Stage\s*(\d+)', name_str, re.IGNORECASE)
                if stage_match:
                    stage_num = stage_match.group(1)
                    print(f"   🔍 从裂缝名称推断: Stage {stage_num}")
                else:
                    print(f"   ℹ️  未找到明确的段信息，可能为单段裂缝")
            
            # 显示时间步信息（再次强调）
            if 'n_time_steps' in fracture_info:
                print(f"\n⏱️  时间步 (来自FractureSet): {fracture_info['n_time_steps']}")
            
            if 'n_elements_max' in fracture_info:
                print(f"🔲 最大单元数: {fracture_info['n_elements_max']}")
            
            if 'n_cells' in fracture_info:
                print(f"📐 每个单元的Cell数: {fracture_info['n_cells']}")


def get_simulation_metadata(file_path: str, sim_path: str) -> Dict[str, Any]:
    """
    读取指定simulation的metadata信息
    
    参数:
        file_path: HDF5文件路径
        sim_path: Simulation的完整路径，如 "Fracture Simulation 03"
    
    返回:
        包含metadata信息的字典
    """
    metadata = {}
    
    with h5py.File(file_path, 'r') as f:
        if sim_path not in f:
            print(f"Simulation路径 {sim_path} 不存在")
            return metadata
        
        # 读取 Metadata/Identification
        ident_path = f"{sim_path}/Metadata/Identification"
        
        if ident_path in f:
            ident_group = f[ident_path]
            
            # 读取Name属性
            if "Name" in ident_group:
                name_data = ident_group["Name"][()]
                if isinstance(name_data, bytes):
                    metadata['Name'] = name_data.decode('utf-8')
                else:
                    metadata['Name'] = str(name_data)
            
            # 读取其他可能的属性
            for key in ident_group.keys():
                if key != "Name":
                    data = ident_group[key][()]
                    if isinstance(data, bytes):
                        metadata[key] = data.decode('utf-8')
                    else:
                        metadata[key] = data
            
            # 读取attributes
            for attr_key, attr_val in ident_group.attrs.items():
                if isinstance(attr_val, bytes):
                    metadata[f"attr_{attr_key}"] = attr_val.decode('utf-8')
                else:
                    metadata[f"attr_{attr_key}"] = attr_val
        
        # 尝试读取其他Metadata信息
        other_metadata = {
            "Geometry": f"{sim_path}/Metadata/Geometry",
            "Simulation": f"{sim_path}/Metadata/Simulation",
            "Results": f"{sim_path}/Metadata/Results",
            "Units": f"{sim_path}/Metadata/Units"
        }
        
        for meta_name, meta_path in other_metadata.items():
            if meta_path in f:
                meta_group = f[meta_path]
                for key in meta_group.keys():
                    try:
                        data = meta_group[key][()]
                        if isinstance(data, bytes):
                            metadata[f"{meta_name}_{key}"] = data.decode('utf-8')
                        else:
                            metadata[f"{meta_name}_{key}"] = data
                    except:
                        pass
                
                for attr_key, attr_val in meta_group.attrs.items():
                    if isinstance(attr_val, bytes):
                        metadata[f"{meta_name}_attr_{attr_key}"] = attr_val.decode('utf-8')
                    else:
                        metadata[f"{meta_name}_attr_{attr_key}"] = attr_val
    
    return metadata


def get_fracture_set_info(file_path: str, sim_path: str) -> Dict[str, Any]:
    """
    读取FractureSet信息，包括裂缝名称、类型、段信息等
    """
    info = {}
    
    with h5py.File(file_path, 'r') as f:
        fracture_set_path = f"{sim_path}/Results/Bulk/UFM/FractureSet"
        
        if fracture_set_path not in f:
            return info
        
        fracture_set = f[fracture_set_path]
        
        # 读取基本属性
        if "ElementCount" in fracture_set:
            element_count = fracture_set["ElementCount"][:]
            info['ElementCount'] = element_count
            info['n_time_steps'] = len(element_count)
        
        # 读取Elements信息
        elements_path = f"{fracture_set_path}/Elements"
        if elements_path in f:
            elements = f[elements_path]
            
            # 获取Points信息
            if "Points" in elements:
                points = elements["Points"]
                if "X" in points:
                    X = points["X"][:]
                    info['n_elements_max'] = X.shape[1] if X.ndim > 1 else X.shape[0]
                    info['n_corners'] = X.shape[2] if X.ndim > 2 else 4
        
        # 读取Cells信息
        cells_path = f"{fracture_set_path}/Elements/Cells"
        if cells_path in f:
            cells = f[cells_path]
            if "WidthProfile" in cells:
                width_profile = cells["WidthProfile"][:]
                info['n_cells'] = width_profile.shape[2] if width_profile.ndim > 2 else 1
                info['has_WidthProfile'] = True
        
        # 读取Mesh信息
        mesh_path = f"{sim_path}/Results/Bulk/UFM/Mesh"
        if mesh_path in f:
            mesh = f[mesh_path]
            if "Stage" in mesh:
                stage_data = mesh["Stage"][:]
                info['Stage'] = stage_data
                if isinstance(stage_data, np.ndarray):
                    info['n_stages'] = len(np.unique(stage_data))
    
    return info


def get_specific_fracture_info(file_path: str, sim_id: int):
    """
    获取特定裂缝的详细信息，包括所有可用的属性和数据
    """
    sim_name = format_sim_name(sim_id)
    
    print(f"\n{'='*80}")
    print(f"详细检查 {sim_name} (ID: {sim_id})")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        if sim_name not in f:
            print(f"{sim_name} 不存在")
            print(f"可用的裂缝:")
            for key in f.keys():
                if key.startswith("Fracture Simulation"):
                    print(f"  - {key}")
            return
        
        # 显示时间步信息
        time_info = get_time_steps_info(file_path, sim_name)
        print(f"⏱️  时间步统计:")
        print(f"   总时间步数: {time_info['n_time_steps']}")
        if time_info['time_values'] is not None:
            print(f"   时间值范围: {time_info['time_values'][0]:.4f} - {time_info['time_values'][-1]:.4f} {time_info['time_unit']}")
            print(f"   时间增量: {np.diff(time_info['time_values'])[0] if len(time_info['time_values']) > 1 else 'N/A'} {time_info['time_unit']}")
        
        def explore_group(group_path, indent=0, max_depth=3):
            """递归探索group内容，限制深度"""
            if indent > max_depth:
                return
            group = f[group_path]
            prefix = "  " * indent
            
            for key in sorted(group.keys()):
                obj = group[key]
                full_path = f"{group_path}/{key}"
                
                if isinstance(obj, h5py.Dataset):
                    # 数据集
                    shape = obj.shape
                    dtype = obj.dtype
                    print(f"{prefix}📄 {key}: shape={shape}, dtype={dtype}")
                    
                    # 如果是小数据集，显示部分值
                    if np.prod(shape) < 10:
                        try:
                            data = obj[()]
                            if isinstance(data, bytes):
                                data = data.decode('utf-8')
                            print(f"{prefix}   └─ 值: {data}")
                        except:
                            pass
                
                elif isinstance(obj, h5py.Group):
                    # 组
                    print(f"{prefix}📁 {key}/")
                    explore_group(full_path, indent + 1, max_depth)
        
        print("\n完整目录结构 (前3层):")
        explore_group(sim_name, 0, 3)
        
        # 特别关注Identification下的Name
        print("\n" + "="*80)
        print("重点关注: Metadata/Identification/Name")
        print("="*80)
        
        ident_path = f"{sim_name}/Metadata/Identification"
        if ident_path in f:
            ident = f[ident_path]
            if "Name" in ident:
                name_data = ident["Name"][()]
                if isinstance(name_data, bytes):
                    name = name_data.decode('utf-8')
                else:
                    name = str(name_data)
                print(f"\n✅ Metadata中的裂缝名称: {name}")
                
                # 读取所有attributes
                for attr_name, attr_val in ident.attrs.items():
                    if isinstance(attr_val, bytes):
                        attr_val = attr_val.decode('utf-8')
                    print(f"   📌 {attr_name}: {attr_val}")
            
            # 列出Identification下的所有内容
            print(f"\n📂 Identification组内容:")
            for key in ident.keys():
                data = ident[key][()]
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                print(f"   {key}: {data}")
        else:
            print("\n❌ 未找到 Metadata/Identification 路径")
            print("   搜索包含'Name'的数据集:")
            
            # 搜索所有包含Name的路径
            def find_name_paths(name, obj):
                if isinstance(obj, h5py.Dataset) and "Name" in name:
                    print(f"   📍 {name}")
            
            f[sim_name].visititems(find_name_paths)


if __name__ == "__main__":
    # 文件路径
    # target = r"D:\git\ufm-hdf5\data\hdf5_file\JY108-7HF_cluster1.h5"
    target = r"D:\git\ufm-hdf5\data\hdf5_file\JY68-4HF.h5"
    
    # 1. 列出所有Fracture Simulation（推荐）
    list_all_fracture_simulations(target, show_details=True)
    
    # 2. 如果需要查看特定裂缝的详细信息，取消下面的注释
    # get_specific_fracture_info(target, sim_id=3)
    # get_specific_fracture_info(target, sim_id=24)
    
    # 3. 如果需要探索完整的HDF5结构，取消下面的注释
    # explore_hdf5_structure(target, max_depth=2)