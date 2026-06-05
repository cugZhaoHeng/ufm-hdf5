import h5py

def inspect_units(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # 假设路径是这样
        target_path = "Fracture Simulation 45/Results/Bulk/UFM/FractureSet/Elements/Temperature"
        
        if target_path in f:
            dset = f[target_path]
            # 查看该数据集的所有属性名称
            print(f"属性列表: {list(dset.attrs.keys())}")
            
            # 尝试直接读取单位
            if 'units' in dset.attrs:
                print(f"单位是: {dset.attrs['units']}")
            elif 'unit' in dset.attrs:
                print(f"单位是: {dset.attrs['unit']}")
            else:
                print("未在属性中找到单位。")
        else:
            print("路径不存在")

# 运行检查
inspect_units(r"D:\git\ufm-hdf5\data\hdf5_file\JY68-4HF.h5")